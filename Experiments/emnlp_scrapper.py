import urllib2
from bs4 import BeautifulSoup

def scrape_emnlp(id):
	url="http://aclweb.org/anthology/D/D"+str(id)+"/"
	home=urllib2.urlopen(url)
	soup=BeautifulSoup(home,"lxml")
	links=soup.find("div",id="content").find_all("p")
	link_array=[]
	i=0
	for i in range(2,len(links)):
		name=links[i].find("a").string+".pdf"
		pdfurl=url+name
		rq=urllib2.Request(pdfurl)
		res=urllib2.urlopen(rq)
		pdf=open("EMNLP/"+name,"wb")
		pdf.write(res.read())
		pdf.close()


x=[15,16]
for y in x:
	scrape_emnlp(y)