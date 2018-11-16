from Bio import SeqIO

file_in ='oma-pairs.txt'
file_out='oma-pairs-viridiplantae.txt'
file_list = 'oma-viridiplantae.txt'

mylist = []
for line in open(file_list, mode='r'):
	mylist.append(line.rstrip())

with open(file_out, 'w+') as f_out:
    for line in open(file_in, mode='r'):
        words = line.split()
        species1 = words[0].rstrip()[0:5]
        species2 = words[1].rstrip()[0:5]
        if ((species1 in mylist)and(species2 in mylist)):
			f_out.write(line)
