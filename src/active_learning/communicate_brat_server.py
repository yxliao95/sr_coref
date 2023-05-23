import logging
import os
import sys
from stat import S_ISDIR

import hydra
import paramiko
import pexpect
from tqdm.auto import tqdm

logger = logging.getLogger()
module_path = os.path.dirname(__file__)
config_path = os.path.join(os.path.dirname(module_path), "config")


class RemoteConnection:
    def __init__(self, hostname, username, password):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(hostname, 22, username, password)
        self.sftp = self.ssh.open_sftp()

    def close_client(self):
        self.ssh.close()
        self.sftp.close()

    def exec_command(self, command_str):
        stdin, stdout, stderr = self.ssh.exec_command(command_str)
        return stdin, stdout, stderr

    def remotepath_join(self, *args):
        # Bug fix for Windows clients, we always use / for remote paths
        return "/".join(args)

    def sftp_walk(self, remotepath):
        # Kindof a stripped down  version of os.walk, implemented for
        # sftp.  Tried running it flat without the yields, but it really
        # chokes on big directories.
        path = remotepath
        files = []
        folders = []
        for f in self.sftp.listdir_attr(remotepath):
            if S_ISDIR(f.st_mode):
                folders.append(f.filename)
            else:
                files.append(f.filename)
        yield path, folders, files
        for folder in folders:
            new_path = self.remotepath_join(remotepath, folder)
            for x in self.sftp_walk(new_path):
                yield x

    def put(self, localfile, remotefile):
        # Copy localfile to remotefile, overwriting or creating as needed.
        self.sftp.put(localfile, remotefile)

    def put_all(self, localpath, remotepath):
        # create parent dir
        self.exec_command(f"mkdir -p {remotepath}")
        # recursively upload a full directory
        os.chdir(os.path.split(localpath)[0])
        base_dir = os.path.split(localpath)[1]
        for path, _, files in tqdm(os.walk(base_dir)):
            try:
                self.sftp.mkdir(self.remotepath_join(remotepath, path))
            except:  # pylint: disable=bare-except
                pass
            for filename in tqdm(files):
                self.put(os.path.join(path, filename), self.remotepath_join(remotepath, path, filename))

    def get(self, remotefile, localfile):
        # Copy remotefile to localfile, overwriting or creating as needed.
        self.sftp.get(remotefile, localfile)

    def get_all(self, remotepath, localpath):
        # Recursively download a full directory
        #
        # For the record, something like this would gennerally be faster:
        # ssh user@host 'tar -cz /source/folder' | tar -xz

        self.sftp.chdir(os.path.split(remotepath)[0])
        base_dir = os.path.split(remotepath)[1]
        try:
            os.mkdir(localpath)
        except FileExistsError:
            pass
        for path, _, files in tqdm(self.sftp_walk(base_dir)):
            try:
                os.mkdir(self.remotepath_join(localpath, path))
            except FileExistsError:
                pass
            for filename in tqdm(files):
                # print(self.remotepath_join(path, filename), os.path.join(localpath, path, filename))
                self.get(self.remotepath_join(path, filename), os.path.join(localpath, path, filename))

    def rename(self, remotepath_old, remotepath_new):
        self.exec_command(f"mv {remotepath_old} {remotepath_new}")


@hydra.main(version_base=None, config_path=config_path, config_name="active_learning")
def main(config):
    brat_output_dir = config.output.brat.unfinished_dir
    brat_server_dir = config.remote_server.brat.data_dir
    base_name_old = os.path.basename(brat_output_dir)
    base_name_new = f"iter_{config.current_iter}"

    connection = RemoteConnection(
        config.remote_server.brat.hostname, config.remote_server.brat.username, config.remote_server.brat.password
    )
    connection.put_all(
        brat_output_dir,
        brat_server_dir,
    )
    connection.rename(os.path.join(brat_server_dir, base_name_old), os.path.join(brat_server_dir, base_name_new))
    connection.close_client()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
