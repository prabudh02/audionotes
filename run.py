import streamlit.cli as stcli
import sys

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", "index2.py", "--server.port=8501", "--server.address=0.0.0.0"]
    sys.exit(stcli.main())