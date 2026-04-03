"""
FT (Fine-tune) Timeline Viewer
Hierarchical view: Session > Loop > Stage > EvoLoop > Events

Run:
    streamlit run rdagent/app/finetune/llm/ui/app.py
"""

import os
from pathlib import Path

import streamlit as st
from streamlit import session_state as state

from rdagent.app.finetune.llm.ui.benchmarks import get_core_metric_score
from rdagent.app.finetune.llm.ui.components import render_session, render_summary
from rdagent.app.finetune.llm.ui.config import ALWAYS_VISIBLE_TYPES, OPTIONAL_TYPES
from rdagent.app.finetune.llm.ui.data_loader import (
    get_summary,
    get_valid_sessions,
    load_ft_session,
)
from rdagent.app.finetune.llm.ui.ft_summary import render_job_summary

DEFAULT_LOG_BASE = "log/"


def validate_path_within_cwd(user_path: Path) -> Path:
    """
    Validate that a user-provided path is within the current working directory.

    Security: This function prevents path traversal attacks by:
    1. Resolving the path to its absolute canonical form
    2. Verifying it's within the CWD boundary using a normalized common prefix
    3. Rejecting paths outside the boundary with ValueError

    Parameters
    ----------
    user_path : Path
        User-provided path to validate

    Returns
    -------
    Path
        Resolved absolute path if valid

    Raises
    ------
    ValueError
        If path is outside the current working directory
    """
    safe_root = Path.cwd().resolve()
    # Expand any user home reference and resolve without requiring the path to exist.
    resolved_path = user_path.expanduser().resolve(strict=False)

    # Ensure the resolved path is absolute and remains within the safe root.
    safe_root_str = str(safe_root)
    resolved_str = str(resolved_path)
    common = os.path.commonpath([safe_root_str, resolved_str])
    if common != safe_root_str:
        raise ValueError("Path is outside the allowed project directory")

    # This will raise ValueError if resolved_path is not within safe_root
    resolved_path.relative_to(safe_root)

    return resolved_path


def get_job_options(base_path: Path) -> list[str]:
    """
    Scan directory and return job options list.
    - "." means standalone tasks in root directory
    - Others are job directory names

    Security: Validates base_path to prevent path traversal attacks.
    Only allows scanning directories within the current working directory.
    """
    options = []
    has_root_tasks = False
    job_dirs = []

    # Security: Validate base_path to prevent path traversal
    try:
        # Use dedicated validation function for path traversal prevention
        base_path_resolved = validate_path_within_cwd(base_path)
    except ValueError:
        # Path is outside the allowed root, reject it.
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
        # Check if standalone task (has __session__ directly)
        if (d / "__session__").exists():
            has_root_tasks = True
        # Check if job directory (subdirs have __session__)
        else:
            try:
                if any((sub / "__session__").exists() for sub in d.iterdir() if sub.is_dir()):
                    job_dirs.append(d.name)
            except PermissionError:
                pass

    # Sort job dirs by name descending (newest first, since names are date-based)
    job_dirs.sort(reverse=True)

    # Add job dirs first, then root tasks at the end
    options.extend(job_dirs)
    if has_root_tasks:
        options.append(". (Current)")

    return options


def main():
    st.set_page_config(layout="wide", page_title="FT Timeline", page_icon="🔬")

    # ========== Sidebar ==========
    with st.sidebar:
        # View mode selection
        view_mode = st.radio("View Mode", ["Job Summary", "Single Task"], horizontal=True)

        st.divider()

        default_log = os.environ.get("FT_LOG_PATH", DEFAULT_LOG_BASE)
        job_folder = default_log  # Initialize for both modes
        selected_types = ALWAYS_VISIBLE_TYPES.copy()  # Initialize for both modes
        is_root_job = False  # Track if viewing root tasks

        if view_mode == "Job Summary":
            # Job Summary mode
            st.header("Job")
            base_folder = st.text_input("Base Folder", value=default_log, key="base_folder_input")
            base_path = Path(base_folder)

            job_options = get_job_options(base_path)
            if job_options:
                selected_job = st.selectbox("Select Job", job_options, key="job_select")
                if selected_job.startswith("."):
                    job_folder = base_folder
                    is_root_job = True
                else:
                    job_folder = str(base_path / selected_job)
                # Save to session_state for Single Task mode
                state.selected_job_folder = job_folder
            else:
                st.warning("No jobs found in this directory")
                job_folder = base_folder

            if st.button("Refresh", type="primary", key="refresh_job"):
                st.rerun()
        else:
            # Single Task mode
            st.header("Session")
            # Use job_folder from Job Summary mode if available
            default_path = getattr(state, "selected_job_folder", default_log)
            log_folder = st.text_input("Log Folder", value=default_path)
            log_path = Path(log_folder)

            sessions = get_valid_sessions(log_path)
            if not sessions:
                st.warning("No valid sessions found")
                return

            selected_session = st.selectbox("Session", sessions)

            if st.button("Load", type="primary") or "session" not in state:
                with st.spinner("Loading..."):
                    state.session = load_ft_session(log_path / selected_session)
                    state.session_name = selected_session

            st.divider()

            # Optional type toggles
            st.subheader("Show More")
            selected_types = ALWAYS_VISIBLE_TYPES.copy()
            for event_type, (label, default) in OPTIONAL_TYPES.items():
                if st.toggle(label, value=default, key=f"toggle_{event_type}"):
                    selected_types.append(event_type)

            st.divider()

            # Display options
            st.subheader("Display Options")
            state.render_markdown = st.toggle("Render Prompts", value=False, key="render_markdown_toggle")

            st.divider()

            # Summary in sidebar
            if "session" in state:
                summary = get_summary(state.session)
                st.subheader("Summary")
                st.metric("Loops", summary.get("loop_count", 0))
                st.metric("LLM Calls", summary.get("llm_call_count", 0))
                success = summary.get("docker_success", 0)
                fail = summary.get("docker_fail", 0)
                st.metric("Docker", f"{success}✓ / {fail}✗")

    # ========== Main Content ==========
    if view_mode == "Job Summary":
        st.title("📊 FT Job Summary")

        # Security: Validate job_folder to prevent path traversal
        # Only allow paths within the base_path directory
        try:
            safe_root = Path(base_path).resolve()
            
            # Additional security: Validate job_folder doesn't contain path traversal sequences
            # This prevents CodeQL path-injection warning
            if ".." in job_folder or job_folder.startswith("/") or job_folder.startswith("\\"):
                st.error("Invalid job folder: Path traversal sequences not allowed")
                st.info("Please select a valid job from the sidebar.")
                return
            
            job_path = Path(job_folder).expanduser().resolve(strict=False)

            # Ensure job_path is within safe_root (prevent path traversal)
            job_path.relative_to(safe_root)

            if job_path.exists():
                render_job_summary(job_path, is_root=is_root_job)
            else:
                st.warning(f"Job folder not found: {job_folder}")
        except ValueError:
            st.error("Invalid job folder path: Must be within base directory")
            st.info("Please select a valid job from the sidebar.")
        except (OSError, RuntimeError) as e:
            st.error(f"Invalid path: {e}")
            st.info("Please select a valid job from the sidebar.")
        return

    # Single Task mode
    st.title("🔬 FT Timeline Viewer")

    if "session" not in state:
        st.info("Select a session and click **Load** to view")
        return

    session = state.session
    summary = get_summary(session)

    # Global info header (Base Model, Datasets, Benchmark) - compact style
    scenario_event = next((e for e in session.init_events if e.type == "scenario"), None)
    dataset_event = next((e for e in session.init_events if e.type == "dataset_selection"), None)

    if scenario_event or dataset_event:
        if scenario_event and hasattr(scenario_event.content, "base_model"):
            st.markdown(f"🧠 **Model:** `{scenario_event.content.base_model}`")
        if dataset_event:
            selected = (
                dataset_event.content.get("selected_datasets", []) if isinstance(dataset_event.content, dict) else []
            )
            if selected:
                st.markdown(f"📂 **Datasets:** `{', '.join(selected)}`")
        if scenario_event and hasattr(scenario_event.content, "target_benchmark"):
            st.markdown(f"🎯 **Benchmark:** `{scenario_event.content.target_benchmark}`")
        # Display baseline benchmark score
        if scenario_event and hasattr(scenario_event.content, "baseline_benchmark_score"):
            baseline = scenario_event.content.baseline_benchmark_score
            if baseline and isinstance(baseline, dict):
                benchmark_name = getattr(scenario_event.content, "target_benchmark", "")
                accuracy_summary = baseline.get("accuracy_summary", {})
                if accuracy_summary:
                    result = get_core_metric_score(benchmark_name, accuracy_summary)
                    if result:
                        metric_name, score, _ = result
                        st.markdown(f"📊 **Baseline:** `{metric_name} = {score:.1f}`")

    # Summary bar
    render_summary(summary)

    st.divider()

    # Hierarchical view
    render_session(session, selected_types)


if __name__ == "__main__":
    main()
