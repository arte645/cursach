from lib.model.api import LlmRag
import os, sys


def set_environment_variables(env_path: str = '../.env',):
    with open(env_path, 'r') as f:
        key, val = f.readline().split()
        os.environ[key] = val

if __name__ == '__main__':
    set_environment_variables('.env')
    
    llm = LlmRag(os.environ['GIGACHAT_API_KEY'], 'preprocessedDocs')
    stop = 0
    while not stop:
        message = input('>>>')
        if not message:
            stop = 1
        else:
            print(llm.invoke(message))

