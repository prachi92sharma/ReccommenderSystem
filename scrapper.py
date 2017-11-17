import urllib2
from bs4 import BeautifulSoup

def scrape_abstract(id,f):
	url="https://nips.cc/Conferences/2017/Schedule?showEvent="+id
	home=urllib2.urlopen(url)
	soup=BeautifulSoup(home,"lxml")
	title=soup.find("div",class_="maincardBody").string
	if soup.find("div",class_="abstractContainer").p==None:
		abstract=soup.find("div",class_="abstractContainer").string
	else:
		abstract=soup.find("div",class_="abstractContainer").p.string
	if title!=None and abstract!=None:
		f.write(title.encode("utf8").strip("\n")+","+abstract.encode("utf8").strip("\n")+"\n")

def scrape_nips(url,f):
	home=urllib2.urlopen(url)
	soup=BeautifulSoup(home,"lxml")
	table=soup.find_all("div",class_="narrower")
	i=0
	for t in table:
		scrape_abstract(t.get("id").split("_")[1],f)

f=open("abstract.txt","a")
years=["2012","2013"]
for y in years:
	print y+" Starting"
	scrape_nips("https://nips.cc/Conferences/"+y+"/Schedule?type=Poster",f)
	print y+" Ending"
f.close()