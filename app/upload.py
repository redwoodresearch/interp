import os
import time
import datetime
import hashlib

import boto3

import streamdb.rrserve

with open("dist/index.js") as f:
    data = f.read()

s3 = boto3.client("s3")

build_time = datetime.datetime.now().strftime("%y-%m-%dT%H:%M")
build_hash = hashlib.sha256(data.encode()).hexdigest()[:16]

branch = os.environ["CIRCLE_BRANCH"]
commit_hash = os.environ["CIRCLE_SHA1"]

key = "/rrinterp-widgets-builds/rrinterp-widgets-%s-%s-%s.js" % (branch.replace("/", "-"), build_time, commit_hash)
print("Uploading to:", key)

streamdb.rrserve.upload_to_rrserve(
    key=key, category="rrinterp-widgets:" + branch, data=data.encode(), content_type="text/javascript"
)
