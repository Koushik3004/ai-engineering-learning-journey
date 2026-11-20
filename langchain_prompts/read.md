python -m venv venv
venv\Scripts\Activate
python -m pip install -r requirements.txt
python -m pip install python-dotenv
streamlit run prompt_ui.py