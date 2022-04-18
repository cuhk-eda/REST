import os

class LogIt:
    def __init__(self, dir):
        print('[LogIt] Init log. Log dir:', dir)
        self.log_dir = dir
        if os.path.exists(dir):
            # os.remove(dir)
            # print('[LogIt] The old log is deleted.')
            print('[LogIt] Append to exisiting log.')

    def log(self, message):
        with open(self.log_dir, 'a') as logf:
            logf.write(message)

    def log_iter(self, iter, dict):
        msg = 'iter ' + str(iter)
        for key in dict:
            msg += ' ' + key + ' ' + str(dict[key])
        msg += '\n'
        self.log(msg)

