import argparse
import yaml
from train_main import train_main


def main():
    parser = argparse.ArgumentParser(description='train a model with your params')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration YAML file.')

    args = parser.parse_args()
    try:
        with open(args.config, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError as e:
        print(e)
        exit(1)
    print('Parsing ......')
    train_main(config)

if __name__ == '__main__':
    main()
    
    


