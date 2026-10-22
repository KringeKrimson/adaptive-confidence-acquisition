import argparse
import json
from .data import prepare


def main(argv=None):
    parser = argparse.ArgumentParser(description='Adaptive Confidence Acquisition')
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--out',required=True)
    p.add_argument('--train',type=int,default=0)
    p.add_argument('--validation',type=int,default=0)
    p.add_argument('--test',type=int,default=32)
    p.add_argument('--seed',type=int,default=4901)
    p.add_argument('--exclude-manifest',help='Exclude inspected pilot queries from every research split')
    p = sub.add_parser('collect')
    p.add_argument('--out',required=True)
    p.add_argument('--stages',nargs='+',choices=['small','strong','ic'],default=['small','strong'])
    p.add_argument('--max-new-tokens',type=int,default=32)
    p.add_argument('--quantization',choices=['4bit','none'],default='4bit')
    p = sub.add_parser('analyze')
    p.add_argument('--out',required=True)
    p.add_argument('--random-seeds',type=int,default=20)
    p = sub.add_parser('summary')
    p.add_argument('--out',required=True)
    args = parser.parse_args(argv)
    options = vars(args).copy()
    command = options.pop('command')
    root = options.pop('out')
    if command == 'prepare':
        manifest = prepare(root,**options)
        print(f'Saved {len(manifest["queries"])} queries to {root}')
    elif command == 'collect':
        from .collect import collect
        collect(root,**options)
    elif command == 'analyze':
        from .analysis import analyze
        report = analyze(root,**options)
        print(json.dumps(report['results'],indent=2))
    else:
        from .collect import pilot_summary
        pilot_summary(root)


if __name__ == '__main__':
    main()
