from Bio import SeqIO

file_in ='oma-seqs.fa'
file_out='oma-seqs-viridiplantae.fa'
file_list = 'oma-viridiplantae.txt'

mylist = []
for line in open(file_list, mode='r'):
	mylist.append(line.rstrip())

with open(file_out, 'w+') as f_out:
    for seq_record in SeqIO.parse(open(file_in, mode='r'), 'fasta'):
        #name, sequence = seq_record.id, str(seq_record.seq)
        species = seq_record.id.rstrip()[0:5]
        if (species in mylist):
			r=SeqIO.write(seq_record, f_out, 'fasta')
			if r!=1: print('Error while writing sequence:  ' + seq_record.id)
