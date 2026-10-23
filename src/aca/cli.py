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
    p.add_argument('--revision',help='Pinned dataset commit')
    p.add_argument('--exclude-manifest',help='Exclude inspected pilot queries from every research split')
    p = sub.add_parser('collect')
    p.add_argument('--out',required=True)
    p.add_argument('--stages',nargs='+',choices=['small','strong','ic'],default=['small','strong'])
    p.add_argument('--splits',nargs='+',choices=['train','validation','test'],default=['train','validation','test'])
    p.add_argument('--max-new-tokens',type=int,default=32)
    p.add_argument('--quantization',choices=['4bit','4bit-fp32','none'],default='4bit-fp32')
    p.add_argument('--small-model',choices=['microsoft/Phi-3-mini-4k-instruct','Qwen/Qwen2.5-1.5B-Instruct'],default='microsoft/Phi-3-mini-4k-instruct')
    p.add_argument('--answer-stop',choices=['eos','first-line'],default='eos')
    p.add_argument('--strong-model',choices=['Qwen/Qwen2.5-7B-Instruct','Qwen/Qwen2.5-14B-Instruct'],
                   default='Qwen/Qwen2.5-7B-Instruct')
    p = sub.add_parser('analyze')
    p.add_argument('--out',required=True)
    p.add_argument('--random-seeds',type=int,default=20)
    p = sub.add_parser('summary')
    p.add_argument('--out',required=True)
    p = sub.add_parser('freeze-methods')
    p.add_argument('--out',required=True)
    p = sub.add_parser('compare-development')
    p.add_argument('--out',required=True)
    p = sub.add_parser('confirmatory-evaluate')
    p.add_argument('--out',required=True)
    p = sub.add_parser('viability')
    p.add_argument('--out',required=True)
    p.add_argument('--protocol',choices=['original','qwen15'],default='original')
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
    elif command == 'viability':
        from .analysis import viability
        if options['protocol']=='qwen15':
            from .prospective import PROTOCOL
            report=viability(root,protocol=PROTOCOL)
        else:
            report=viability(root)
        print(json.dumps(report,indent=2))
    elif command == 'freeze-methods':
        from .frozen import freeze_methods
        from .storage import fingerprint
        report=freeze_methods(root)
        print('Frozen methods:',fingerprint(report),'operating points:',len(report['selected']))
    elif command == 'compare-development':
        from .frozen import compare_development
        print(json.dumps(compare_development(root),indent=2))
    elif command == 'confirmatory-evaluate':
        from .confirmatory import evaluate
        report=evaluate(root)
        print(json.dumps(report,indent=2))
    else:
        from .collect import pilot_summary
        pilot_summary(root)


if __name__ == '__main__':
    main()
