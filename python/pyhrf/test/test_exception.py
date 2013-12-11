import sys


raise Exception('The first evil exception')
print sys.argv


def main():
    raise Exception('A not so evil exception')

if __name__ == '__main__':
    main()
