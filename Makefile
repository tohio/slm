python3 -c "
import boto3, os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

s3 = boto3.client('s3')
bucket = os.environ['S3_BUCKET']
prefix = os.environ.get('S3_PREFIX', 'slm/data')
tokenizer_dir = Path('data/tokenizer')

for f in tokenizer_dir.iterdir():
    key = f'{prefix}/tokenizer/{f.name}'
    print(f'Uploading {f.name} → s3://{bucket}/{key}')
    s3.upload_file(str(f), bucket, key)
print('Done')
"