import boto3
from botocore.exceptions import ClientError
import os
from glob import glob
import logging
import sys
sys.path.append("/home/yyf/.aws/")
import credentials


def get_client():
        s3 = boto3.resource(service_name="s3",
                      aws_access_key_id=credentials.aws_access_key_id,
                      aws_secret_access_key=credentials.aws_secret_access_key)
        return s3


def check_exists(s3, bucket_name, stim_name):
    try:
        s3.Object(bucket_name,stim_name).load()
        return True
    except ClientError as e:
        if (e.response['Error']['Code'] == "404"):
            return False
        else:
            print('Something else has gone wrong with {}'.format(stim_name))

def main():
    bucket = "gestalt-scenes"
    s3 = get_client()
    b = s3.Bucket(bucket)

    b.Acl().put(ACL="public-read")
    root_path = "/om/user/yyf/CommonFate/scenes"
    data_path = root_path + "/test_ground_truth/**/*/*/*" # Upload PNGs
    overwrite = True
    for file_path in glob(data_path):
        if "." in file_path:
            target = file_path.split(root_path)[1][1:]
            if check_exists(s3, bucket, target) and not overwrite:
                print(target + " exists. Skipping")
                continue

            print(target)
            image = Image.open(file_path)
            s3.Object(bucket, target).put(Body=open(file_path,'rb')) ## upload stimuli
            s3.Object(bucket, target).Acl().put(ACL='public-read') ## set access controls

    data_path = root_path + "/test_ground_truth/**/*/*" # Upload everything else
    for file_path in glob(data_path):
        if "." in file_path:

            target = file_path.split(root_path)[1][1:]
            if check_exists(s3, bucket, target) and not overwrite:
                print(target + " exists. Skipping")
                continue
            s3.Object(bucket, target).put(Body=open(file_path,'rb')) ## upload stimuli
            s3.Object(bucket, target).Acl().put(ACL='public-read') ## set access controls
            print(target)

if __name__=="__main__":
    main()
