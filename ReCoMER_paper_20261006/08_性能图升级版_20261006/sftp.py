# -*- coding: utf-8 -*-
"""本地↔服务器文件传输（paramiko SFTP）。"""
import sys
import paramiko

HOST, PORT, USER, PWD = "connect.westd.seetacloud.com", 30558, "root", "7jDXIOsFQDX5"


def sftp():
    t = paramiko.Transport((HOST, PORT))
    t.connect(username=USER, password=PWD)
    s = paramiko.SFTPClient.from_transport(t)
    return t, s


if __name__ == "__main__":
    mode = sys.argv[1]            # up / down
    local, remote = sys.argv[2], sys.argv[3]
    t, s = sftp()
    try:
        if mode == "up":
            s.put(local, remote)
            print("uploaded", local, "->", remote)
        else:
            s.get(remote, local)
            print("downloaded", remote, "->", local)
    finally:
        s.close(); t.close()
