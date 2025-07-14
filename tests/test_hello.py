import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from hello import main

def test_main(capsys):
    main()
    captured = capsys.readouterr()
    assert captured.out.strip() == "Hello, World!"
