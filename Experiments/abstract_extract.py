import os
from os import listdir
import urllib2
from bs4 import BeautifulSoup

def listfiles():
	url="http://aclweb.org/anthology/D/D14/"
	home=urllib2.urlopen(url)
	soup=BeautifulSoup(home,"lxml")
	links=soup.find("div",id="content").find_all("p")
	link_array=[]
	i=0
	for i in range(2,len(links)):
		name=links[i].find("a").string+".pdf"
		filename="EMNLP/"+name
		newpath = "EMNLP_TEXT/"+links[i].find("a").string+".txt"
		print filename
		os.system("pdftotext {0} {1}".format(filename,newpath))
		i=i+1
		if i > 10:
	 	   	break



listfiles()