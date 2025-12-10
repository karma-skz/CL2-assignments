from isanlp_rst.parser import Parser
import asyncio
from isanlp_rst.rstviewer.main import rs3topng_async

version = 'rstdt'  # Choose from {'gumrrg', 'rstdt', 'rstreebank'}
parser = Parser(hf_model_name='tchewik/isanlp_rst_v3', hf_model_version=version, cuda_device=0)

with open('paragraph.txt', 'r', encoding='utf-8') as fh:
	lines = [ln.strip() for ln in fh if ln.strip()]
for i, line in enumerate(lines, start=1):
	if '. ' in line and line.split('. ', 1)[0].isdigit():
		text = line.split('. ', 1)[1]
	else:
		text = line

	res = parser(text)
	root = res['rst'][0]
	rs3_name = f'{i}.rs3'
	png_name = f'{i}.png'
	root.to_rs3(rs3_name)
	asyncio.run(rs3topng_async(rs3_name, png_name))
