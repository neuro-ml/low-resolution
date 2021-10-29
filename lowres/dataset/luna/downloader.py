import os
import argparse

from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('destination', type=str,
                        help='Absolute path to the folder in which the dataset will be downloaded.')

    args = parser.parse_args()

    set1 = [
        'annotations.csv',
        'candidates.csv',
        'candidates_V2.zip',
        'evaluationScript.zip',
        'sampleSubmission.csv',
        'seg-lungs-LUNA16.zip',
        'subset0.zip',
        'subset1.zip',
        'subset2.zip',
        'subset3.zip',
        'subset4.zip',
        'subset5.zip',
        'subset6.zip',
    ]

    set2 = [
        'subset7.zip',
        'subset8.zip',
        'subset9.zip',
    ]

    annotations = [
        'LIDC-XML-only.zip',
    ]

    for file in set1:
        path = Path(args.destination) / file
        if not path.exists():
            os.system(f'wget -O {path} -t inf https://zenodo.org/record/2604219/files/{file}?download=1')
        else:
            print(f'{file} already exists')

    for file in set2:
        path = Path(args.destination) / file
        if not path.exists():
            os.system(f'wget -O {path} -t inf https://zenodo.org/record/2596479/files/{file}?download=1')
        else:
            print(f'{file} already exists')

    for file in annotations:
        path = Path(args.destination) / file

        os.system(
            f'wget -O {path} https://wiki.cancerimagingarchive.net/download/attachments/1966254/LIDC-XML-only.zip?'
            'version=1&modificationDate=1530215018015&api=v2'
        )
