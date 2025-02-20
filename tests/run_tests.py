import pytest
import sys
import os

def main():
    # Add the project root directory to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    # Run tests
    pytest.main(['-v', 'tests'])

if __name__ == '__main__':
    main() 