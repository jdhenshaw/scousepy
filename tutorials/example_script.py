from scousepy import scouse
from scousepy.io import output_ascii_indiv

datadirectory = './' # path to whereever you put the data
outputdir = './' # path to wherever you would like scouse to put the output
filename = 'myfitsfile' # the filename of the data, without the '.fits' extension - e.g. 'n2h+10_37'

# config files can be found in outputdir/config_files/ and can be updated there
config_file=scouse.run_setup(filename, datadirectory, outputdir=outputdir)

# running scouse
s = scouse.stage_1(config=config_file, interactive=True) # GUI based 
s = scouse.stage_2(config=config_file, refit=False)
s = scouse.stage_3(config=config_file)
s = scouse.stage_4(config=config_file, bitesize=True)

#output_ascii_indiv(s, outputdir)
