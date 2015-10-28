import csv


train_path = "datasets/train_ip_normalized.csv"
test_path = "datasets/test_ip_normalized.csv"
cookie_path = "datasets/cookie_ip_normalized.csv"

dev_features = ['drawbridge_handle', 'device_id', 'device_type', 'device_os',
                'country', 'anonymous_c0', 'anonymous_c1', 'anonymous_c2',
                'anonymous_5', 'anonymous_6', 'anonymous_7']

cookie_features = ['drawbridge_handle', 'cookie_id', 'computer_os_type',
                   'computer_browser_version', 'country', 'anonymous_c0',
                   'anonymous_c1', 'anonymous_c2', 'anonymous_5',
                   'anonymous_6', 'anonymous_7']

features = ['label', 'device_id', 'cookie_id', 'device_type', 'computer_os_type',
            'device_os', 'computer_browser_version', 'dev_country', 'cookie_country',
            'dev_anonymous_c0', 'cookie_anonymous_c0', 'dev_anonymous_c1', 'cookie_anonymous_c1',
            'dev_anonymous_c2', 'cookie_anonymous_c2', 'dev_anonymous_5', 'cookie_anonymous_5',
            'dev_anonymous_6', 'cookie_anonymous_6', 'dev_anonymous_7', 'cookie_anonymous_7',
            'common_ips', 'device_ips', 'cookie_ips', 'common_anonymous_1', 'common_anonymous_2',
            'common_anonymous_3', 'common_anonymous_4', 'common_anonymous_5']

train_output = open('datasets/train.csv', 'wb')
test_output = open('datasets/test.csv', 'wb')

open_file_train = csv.writer(train_output)
open_file_test = csv.writer(test_output)

open_file_train.writerow(features)
open_file_test.writerow(features)


def process(devices, cookies, training):
    for t1, dev in enumerate(open(devices)):
        dev = dev.strip().split(',')
        if t1 == 0:
            continue
        for t2, cookie in enumerate(open(cookies)):
            cookie = cookie.strip().split(',')
            if t2 == 0:
                continue
            if training == 1 and cookie[0] == '-1':
                continue
            if dev[0] == cookie[0] and training == 1:
                y = 1
            else:
                y = 0
            dev_id = dev[1]
            cookie_id = cookie[1]
            x = []
            for i in range(2, 11):
                x.append(dev[i])
                x.append(cookie[i])
            dev_ip_all = map(lambda i: i.split(' '), dev[11].strip().split('|'))
            cookie_ip_all = map(lambda i: i.split(' '), cookie[11].strip().split('|'))
            dev_ips = map(lambda i: i[0], dev_ip_all)
            cookie_ips = map(lambda i: i[0], cookie_ip_all)
            common_ips = set(dev_ips).intersection(cookie_ips)
            common_freq = sum([min(int(i[1]), int(j[1])) for i, j in zip(dev_ip_all, cookie_ip_all) if i[0] in common_ips])
            common_anonymous_1 = sum([min(int(i[2]), int(j[2])) for i, j in zip(dev_ip_all, cookie_ip_all) if i[0] in common_ips])
            common_anonymous_2 = sum([min(int(i[3]), int(j[3])) for i, j in zip(dev_ip_all, cookie_ip_all) if i[0] in common_ips])
            common_anonymous_3 = sum([min(int(i[4]), int(j[4])) for i, j in zip(dev_ip_all, cookie_ip_all) if i[0] in common_ips])
            common_anonymous_4 = sum([min(int(i[5]), int(j[5])) for i, j in zip(dev_ip_all, cookie_ip_all) if i[0] in common_ips])
            common_anonymous_5 = sum([min(int(i[6]), int(j[6])) for i, j in zip(dev_ip_all, cookie_ip_all) if i[0] in common_ips])
            x += [len(common_ips), len(dev_ips), len(cookie_ips), common_anonymous_1, common_anonymous_2,
                  common_anonymous_3, common_anonymous_4, common_anonymous_5]
            yield dev_id, cookie_id, x, y, len(common_ips)


if __name__ == '__main__':
    rows = []
    count = 0
    for t, (dev_id, cookie_id, x, y, common) in enumerate(process(train_path, cookie_path, 1)):
        if common > 0 or y == 1:
            row = [y, dev_id, cookie_id] + x
            rows.append(row)
            count += 1
            if count % 10000 == 0 and count > 0:
                print count
                open_file_train.writerows(rows)
                rows = []
    open_file_train.writerows(rows)
    train_output.close()

    rows = []
    count = []
    for t, (dev_id, cookie_id, x, y, common) in enumerate(process(test_path, cookie_path, 0)):
        if common > 0:
            row = [y, dev_id, cookie_id] + x
            rows.append(row)
            count += 1
            if count % 10000 == 0 and count > 0:
                print count
                open_file_test.writerows(rows)
                rows = []
    open_file_test.writerows(rows)
    test_output.close()
