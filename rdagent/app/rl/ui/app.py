"""
RL Post-training Timeline Viewer
Hierarchical view: Session > Loop > Stage > Events

Run:
    streamlit run rdagent/app/rl/ui/app.py
"""

import os
from pathlib import Path

import streamlit as st
from streamlit import session_state as state

from rdagent.app.rl.ui.components import render_session, render_summary
from rdagent.app.rl.ui.config import ALWAYS_VISIBLE_TYPES, OPTIONAL_TYPES
from rdagent.app.rl.ui.data_loader import get_summary, get_valid_sessions, load_session
from rdagent.app.rl.ui.rl_summary import render_job_summary

DEFAULT_LOG_BASE = "log/"


def _safe_resolve(user_input: str | None, safe_root: Path) -> Path:
    """
    Resolve user path relative to safe_root; raise ValueError if it escapes.

    Security: This function prevents path traversal attacks by:
    1. Rejecting null bytes in user input
    2. Rejecting Windows drive letters (C:\, D:\, etc.)
    3. Rejecting absolute paths
    4. Normalizing path to remove .. traversal attempts
    5. Validating resolved path is within safe_root using a realpath-based check

    All user-provided paths are validated before filesystem access.
    """
    # Treat the provided safe_root as trusted and canonicalize it once.
    safe_root = safe_root.expanduser().resolve()

    # Empty input maps to the safe root directory.
    if not user_input:
        return safe_root

    # Security check 1: Reject null bytes (path truncation attack)
    if "\x00" in user_input:
        raise ValueError("Invalid path: contains null byte")

    try:
        # Security check 2: Normalize path to resolve .. and . components
        normalized = os.path.normpath(user_input.strip())

        # Security check 3: Reject Windows drive letters (C:\, D:\, etc.)
        drive, _ = os.path.splitdrive(normalized)
        if drive:
            raise ValueError("Absolute paths with drive letters are not allowed")

        # Security check 4: Reject absolute paths (/, //server/share, etc.)
        if os.path.isabs(normalized):
            raise ValueError("Absolute paths are not allowed")

        # Security check 5: Build candidate path under safe_root and fully resolve it.
        joined = os.path.join(str(safe_root), normalized)
        resolved_candidate = os.path.realpath(joined)

        # Security check 6: Validate candidate is within safe_root (prevent path traversal)
        candidate_path = Path(resolved_candidate)
        candidate_path.relative_to(safe_root)

        return candidate_path
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid path outside of allowed root: {user_input}") from exc


def get_job_options(base_path: Path) -> list[str]:
    """
    Scan directory and return job options list.
    
    Security: Validates base_path to prevent path traversal attacks.
    Only allows scanning directories within the current working directory.
    """
    options = []
    has_root_tasks = False
    job_dirs = []

    # Security fix: Validate base_path to prevent path traversal
    try:
        base_path_resolved = base_path.resolve(strict=False)
        cwd_resolved = Path.cwd().resolve()
        
        # Ensure base_path is within current working directory
        try:
            base_path_resolved.relative_to(cwd_resolved)
        except ValueError:
            # Path is outside CWD, reject it
            st.error("Invalid log base path: Must be within project directory")
            return options
    except (OSError, RuntimeError) as e:
        st.error(f"Invalid path: {e}")
        return options

    if not base_path_resolved.exists():
        return options

    for d in base_path_resolved.iterdir():
        if not d.is_dir():
            continue
        if (d / "__session__").exists():
            has_root_tasks = True
        else:
            try:
                if any((sub / "__session__").exists() for sub in d.iterdir() if sub.is_dir()):
                    job_dirs.append(d.name)
            except PermissionError:
                pass

    job_dirs.sort(reverse=True)
    options.extend(job_dirs)
    if has_root_tasks:
        options.append(". (Current)")

    return options


def main():
    st.set_page_config(layout="wide", page_title="RL Timeline", page_icon="🤖")

    with st.sidebar:
        view_mode = st.radio("View Mode", ["Job Summary", "Single Task"], horizontal=True)
        st.divider()

        default_log = os.environ.get("RL_LOG_PATH", DEFAULT_LOG_BASE)
        safe_root = Path(default_log).expanduser().resolve()
        job_folder = str(safe_root)
        selected_types = ALWAYS_VISIBLE_TYPES.copy()
        is_root_job = False

        if view_mode == "Job Summary":
            st.header("Job")
            base_folder = st.text_input("Base Folder", value=default_log, key="base_folder_input")
            try:
                base_path = _safe_resolve(base_folder, safe_root)
            except ValueError as e:
                st.error(str(e))
                return

            job_options = get_job_options(base_path)
            if job_options:
                selected_job = st.selectbox("Select Job", job_options, key="job_select")
                if selected_job.startswith("."):
                    job_folder = str(base_path)
                    is_root_job = True
                else:
                    job_folder = str(base_path / selected_job)
                state.selected_job_folder = job_folder
            else:
                st.warning("No jobs found in this directory")
                job_folder = str(base_path)

            if st.button("Refresh", type="primary", key="refresh_job"):
                st.rerun()
        else:
            st.header("Session")
            default_path = getattr(state, "selected_job_folder", default_log)
            log_folder = st.text_input("Log Folder", value=default_path)
            try:
                log_path = _safe_resolve(log_folder, safe_root)
            except ValueError as e:
                st.error(str(e))
                return

            sessions = get_valid_sessions(log_path)
            if not sessions:
                st.warning("No valid sessions found")
                return

            selected_session = st.selectbox("Session", sessions)

            if st.button("Load", type="primary") or "session" not in state:
                with st.spinner("Loading..."):
                    state.session = load_session(log_path / selected_session)
                    state.session_name = selected_session

            st.divider()

            st.subheader("Show More")
            selected_types = ALWAYS_VISIBLE_TYPES.copy()
            for event_type, (label, default) in OPTIONAL_TYPES.items():
                if st.toggle(label, value=default, key=f"toggle_{event_type}"):
                    selected_types.append(event_type)

            st.divider()

            if "session" in state:
                summary = get_summary(state.session)
                st.subheader("Summary")
                st.metric("Loops", summary.get("loop_count", 0))
                st.metric("LLM Calls", summary.get("llm_call_count", 0))
                success = summary.get("docker_success", 0)
                fail = summary.get("docker_fail", 0)
                st.metric("Docker", f"{success}✓ / {fail}✗")

    if view_mode == "Job Summary":
        st.title("📊 RL Job Summary")
        try:
            job_path = _safe_resolve(job_folder, safe_root)
        except ValueError as e:
            st.warning(str(e))
            return
        if job_path.exists():
            render_job_summary(job_path, safe_root, is_root=is_root_job)
        else:
            st.warning(f"Job folder not found: {job_folder}")
        return

    st.title("🤖 RL Timeline Viewer")

    if "session" not in state:
        st.info("Select a session and click **Load** to view")
        return

    session = state.session
    summary = get_summary(session)
    render_summary(summary)
    st.divider()
    render_session(session, selected_types)


if __name__ == "__main__":
    main()
