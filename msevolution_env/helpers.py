import csv


class CSVLogger(object):
    def __init__(self, outfile):
        self.file = outfile
        self._log_dict = {}
        self._log_archive = []

    def set_value(self, key, value):
        self._log_dict[key] = value

    def set_values(self, value_dict):
        self._log_dict.update(value_dict)

    def write_log(self):
        # get log-file keys
        csv_keys = set()
        if self.file.exists():
            with open(self.file, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                try:
                    csv_keys = set(next(reader))
                except Exception:
                    pass
        # rewrite csv if unseen keys
        if csv_keys < set(self._log_dict.keys()):
            with open(self.file, 'w+', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, self._log_dict.keys())
                writer.writeheader()
                writer.writerows(self._log_archive)
                writer.writerow(self._log_dict)
        # append csv if no unseen keys
        else:
            with open(self.file, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, self._log_dict.keys())
                writer.writerow(self._log_dict)

        self._log_archive.append(self._log_dict)
        self._log_dict = {k: None for k in self._log_dict.keys()}
