#!/bin/bash

run_streamlit_app() {
    local app_dir=$1
    cd "$app_dir" || return

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    pip install streamlit
    nohup streamlit run app.py >/dev/null 2>&1 &
    deactivate
    cd ..
}

cd tasks/text || exit

run_streamlit_app "autocomplete"
run_streamlit_app "profinity_masker"
run_streamlit_app "speech2text"
run_streamlit_app "translation"
run_streamlit_app "PII"
run_streamlit_app "autocorrect"
