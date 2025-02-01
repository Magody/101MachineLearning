import re

pattern = r'"\/watch\?v=([A-z\d_\-]+)(?:&amp;t=\d+s)?">'



file = open("./input.txt", "r")
lines = file.readlines()
file.close()

urls = set()

for line in lines:
    match = re.findall(pattern, line)

    for url_m in match:
        url = url_m.strip()
        urls.add(url+"\n")



file_urls = open("videos_ids.txt", "w")
file_urls.writelines(urls)
file_urls.close()


