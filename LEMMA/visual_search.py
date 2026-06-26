from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
import re
import json
import pyautogui
import os
import csv


imgbed_root="https://raw.githubusercontent.com/fan19-hub/LEMMA/main/"
options_ = webdriver.ChromeOptions()     # Find the chromederver suitable for your chrome version here: https://googlechromelabs.github.io/chrome-for-testing/#stable, put it under the same directory as this script
options_.add_experimental_option('excludeSwitches', ['enable-logging'])
options_.add_argument("--disable-blink-features=AutomationControlled")  # 隐藏自动化标识
options_.add_experimental_option("excludeSwitches", ["enable-automation"])  # 排除自动化开关
options_.add_experimental_option('useAutomationExtension', False)

driver = webdriver.Chrome(options=options_)


untrusted_sources={"www.reddit.com","www.weibo.com","twitter.com","www.tiktok.com","www.douyin.com","www.instagram.com","www.taobao.com","www.jd.com","www.amazon.com","www.ebay.com","www.imdb.com","www.douban.com","steamcommunity.com","m.ixigua.com","www.bilibili.com","www.netflix.com",}


def source_filter(results):
    global untrusted_sources
    if results == None or results == []:
        return []
    for result in results:
        link=result['href']
        domain = re.search('https?://([A-Za-z_0-9.-]+).*', link).group(1)
        if domain in untrusted_sources:
            results.remove(result)
    return results


def visual_search(source, original_post, is_url=True, max_items = 5):
    try:
        driver.get('https://www.google.com/imghp')
    except:
        try: driver.quit()
        except: pass
        driver = webdriver.Chrome()
        driver.get('https://www.google.com/imghp')
    sleep(1)
    button = driver.find_element(By.CSS_SELECTOR, "div.nDcEnd")
    button.click()
    sleep(5)

    # Get the image
    if is_url:
        # use the image url
        if "http" not in source:
            source=imgbed_root+source
        driver.find_element(By.CSS_SELECTOR, "input.cB9M7").send_keys(source)
        search_button=driver.find_element(By.CSS_SELECTOR, "div.Qwbd3")
        search_button.click()
    else:
        # upload the image
        image_path=os.path.abspath(source)
        pyautogui.typewrite(image_path)
        sleep(3)
        pyautogui.press('enter')
        pyautogui.press('enter')
        sleep(5)
        driver.find_element(By.CSS_SELECTOR, 'input[type="file"]').send_keys(source)
        print("111111")
        sleep(40)


    # exact_search result page
    results=driver.find_elements(By.CSS_SELECTOR, "a.LBcIee")
    print(len(results))
    titles_set = driver.find_elements(By.CSS_SELECTOR,"span.Yt787")
    print(len(titles_set))
    search_results=[]
    for result,titles in zip(results,titles_set):
        link = result.get_attribute('href')
        if "google.com" in link:
            continue
        title = titles.text
        search_results.append({"title":title, "href":link})
    search_results = source_filter(search_results)
    # print(search_results)
    return_list = []
    for search_result in search_results:
        title = search_result['title']
        if title !="":
            # "source":urlparse(link).hostname
            return_list.append("Title: " + title.replace("来源","Source"))
    if return_list==[]:
        driver.quit()
        return "Nothing found"
    retrieved_text = "Image occurs in: " + json.dumps(return_list[:max_items],ensure_ascii=False)
    driver.quit()
    return retrieved_text


# with open("FAKEDDIT.json",'r',encoding="UTF-8") as file:
#     data = json.load(file)
# with open("twitter.json", 'r', encoding="UTF-8") as file:
#     data = json.load(file)
with open("dataset_items_test_filtered.json", 'r', encoding="UTF-8") as file:
    data = json.load(file)

# url_set = []
# for x in range(526,len(data)):
x = 319
for item in list(data.values())[320:]:
    # url = "https://raw.githubusrcontent.com/fan19-hub/LEMMA/main/" + data[x]['image_url']
    url = "C:/Users/26810/PycharmProjects/pythonProject1/" + item['image_path']
    url = url.replace('/', '\\')
    print(url)
    x = x+1
    if os.path.exists(url):
        text = item['caption']
        print(f"现在是第{x}条数据")
        visual_retrieved_text = visual_search(url, text, is_url=False)

        with open("extra_picture_knowledge_LEMMA_MR2.csv", 'a', encoding='utf-8') as file2:
            writer = csv.writer(file2)
            writer.writerow([x, visual_retrieved_text])
        print(visual_retrieved_text)

    # break
