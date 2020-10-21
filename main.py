import config


def print_hi(name):
    print(f'Hi, {name}+{config.age}')  # Press Alt+B to toggle the breakpoint.


if __name__ == '__main__':
    print_hi(config.name)
