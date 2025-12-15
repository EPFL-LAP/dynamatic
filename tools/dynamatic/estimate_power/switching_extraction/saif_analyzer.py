# This file defines the analyzer used to get detailed switching activity results

import re
import logging
from datetime import datetime
from string import Template
import os

#===----------------------------------------------------------------------===#
# Setup logger
#===----------------------------------------------------------------------===#
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#===----------------------------------------------------------------------===#
# Environment Settings
#===----------------------------------------------------------------------===#
# template_folder = "/home/jianliu/Projects/switching_estimation/SAIF_templates"
# result_folder = "/home/jianliu/Projects/switching_estimation/SAIF_results"

# Define regex type
RE_TYPE = type(re.compile(''))

#===----------------------------------------------------------------------===#
# Class/Function Definition
#===----------------------------------------------------------------------===#
def get_date():
    """
        This function will get the running date of the script in isoformat
    """
    current_date = datetime.now()
    str_current_data = current_date.strftime("%a %b %d %H:%M:%S %Y")
    
    return str_current_data

# Reload string.Template method
class NewTemplate(Template):
    delimiter = '%'

class NetInstance(object):
    """
        NetInstance class used to store information relates to a net in saif
    """
    
    def __init__(self, 
                 net_name, 
                 bit_index, 
                 T0,
                 T1,
                 TX,
                 TC,
                 IG
                 ):
        
        # Define net variable
        self.net_name = net_name
        self.bit_index = bit_index
        self.T0 = T0
        self.T1 = T1
        self.TX = TX
        self.TC = TC
        self.IG = IG
        
        self.final_string_name = self.net_name + "\[" + str(self.bit_index) + "\]"
        
    def print_detail(self):
        final_string = "("
        
        # Add channel_name
        final_string += self.net_name
        
        # Add bit_index
        if (self.bit_index != ""):
            final_string += ("\[" + str(self.bit_index) + "\] ")
        else:
            final_string += " "
        
        # Add TO
        final_string += ("(T0 " + str(self.T0) + ") ")
        
        # Add T1
        final_string += ("(T1 " + str(self.T1) + ") ")
        
        # Add TX
        final_string += ("(TX " + str(self.TX) + ") ")
        
        # Add TC
        final_string += ("(TC " + str(self.TC) + ") ")
        
        # Add IG
        final_string += ("(IG " + str(self.IG) + ")")
        final_string += ") "
        
        print("\t{}".format(final_string))
        
    def get_final_string(self):
        final_string = "\t\t\t\t("
        
        # Add channel_name
        final_string += self.net_name
        
        # Add bit_index
        if (self.bit_index != ""):
            final_string += ("\[" + str(self.bit_index) + "\] ")
        else:
            final_string += " "
        
        # Add TO
        final_string += ("(T0 " + str(self.T0) + ") ")
        
        # Add T1
        final_string += ("(T1 " + str(self.T1) + ") ")
        
        # Add TX
        final_string += ("(TX " + str(self.TX) + ") ")
        
        # Add TC
        final_string += ("(TC " + str(self.TC) + ") ")
        
        # Add IG
        final_string += ("(IG " + str(self.IG) + ")")
        final_string += ") "
        
        return final_string
        
    def change_TC(self, new_update):
        self.TC = new_update
    
    def update_TC(self, new_update):
        self.TC += new_update
        

class SAIF(object):
    """
        SAIF class for parsing the saif file
    """
    
    SAIF_KEY_WORDS = set((
        "SAIFILE",
        "SAIFVERSION",
        "DIRECTION",
        "DESIGN",
        "DATE",
        "VENDOR",
        "PROGRAM_NAME",
        "PROGRAM_VERSION",
        "DIVIDER",
        "TIMESCALE",
        "DURATION",
        "INSTANCE",
        "NET",
        ")"
    ))
    
    base_saif_template = """(SAIFILE
    (SAIFVERSION "2.0")
    (DIRECTION "backward")
    (DESIGN )
    (DATE "%{date}")
    (VENDOR "Mentor Graphics")
    (PROGRAM_NAME "vsim")
    (PROGRAM_VERSION "10.7b_1")
    (DIVIDER /)
    (TIMESCALE 1 fs)
    (DURATION %{duration})
    (INSTANCE %{example_name}_tb
        (INSTANCE duv
            (NET\n%{net_list}\n\t\t)
        )
    )
    )
    """
    
    def __init__(self,
        saif_ori_path = None,
        saif_result_path = None,
        example_name = None
    ):
        # Generate the file location
        self.saif_ori_file_path = saif_ori_path + "/" + example_name + "_pre_all.saif"
        self.saif_result_file_path = saif_result_path + "/" + example_name + "_pre_all_out.saif"
        
        # Define storing structure
        ## Format {channel_name : {bit_index : net_instance_class}}
        self.channel_instances = {}
        self.duration = 0
        self.channel_order = []
        self.example_name = example_name
        
        # Parse the input file
        self.parse_saif_file()
        
    def parse_saif_file(self):
        
        # Parse the input file
        with open(self.saif_ori_file_path, "r") as f:
            while True:
                line = f.readline()
                
                # If end of file
                if line == '':
                    break
                
                line = line.strip().split(" ")
                
                # Check content
                line_start = line[0].strip("(")
                if (line_start == "DURATION"):
                    self.duration = int(line[-1].strip(")"))
                    continue
                elif (line_start not in self.SAIF_KEY_WORDS):
                    # Net instance detected
                    ## Get channel name
                    line_con = line_start.split("\[")
                    channel_name = line_con[0]
                    if (len(line_con) > 1):
                        bit_index = int(line_con[-1].strip("\]"))
                    else:
                        bit_index = ""
                    
                    #! Testing
                    # print(line)
                    
                    ## Get TO
                    tmp_t0 = int(line[2].strip(")"))
                    
                    ## Get T1
                    tmp_t1 = int(line[4].strip(")"))
                    
                    ## Get TX
                    tmp_tx = int(line[6].strip(")"))
                    
                    ## Get TC
                    tmp_tc = int(line[8].strip(")"))
                    
                    ## Get IG
                    tmp_ig = int(line[10].strip("))"))
                    
                    # Create the corresponding instance
                    if (channel_name not in self.channel_order):
                        self.channel_order.append(channel_name)
                        
                    if (channel_name not in self.channel_instances.keys()):
                        self.channel_instances[channel_name] = {}
                        
                    self.channel_instances[channel_name][bit_index] = NetInstance(channel_name, 
                                                                                    bit_index, 
                                                                                    tmp_t0,
                                                                                    tmp_t1,
                                                                                    tmp_tx,
                                                                                    tmp_tc,
                                                                                    tmp_ig)
                        
    def target_file_generation(self, template_file, substitute_dict, target_path):
        """
            This file will generate the desired file based on the given template,
            replacement dictionary, and move the file to the desired locatino
        """
        # Write the new file
        print(target_path)
        with open(target_path, "w") as f:
            # Substitute the corresponding location in the template file
            s = NewTemplate(template_file)
            f.write(s.substitute(substitute_dict))
            f.close()
            
    def update_net_TC(self, channel_name, bit_index, value):
        """
            This function add the input number to the TC of the sepecified net
        """
        self.channel_instances[channel_name][bit_index].update_TC(value)
    
    def change_net_TC(self, channel_name, bit_index, value):
        """ 
            This function change the TC value for the specified net
        """
        self.channel_instances[channel_name][bit_index].change_TC(value)
    
    def write_out_saif(self):
        """
            Thif function write out the modified saif file
        """
        # Get the netlist string
        net_list = ""
        for bit_dict in self.channel_instances.values():
            for net_instance in bit_dict.values():
                net_list += (net_instance.get_final_string() + "\n")
        
        # Get substitute dict
        saif_dict = {
            'date' : get_date(),
            'duration' : self.duration,
            'example_name': self.example_name,
            'net_list' : net_list
        }
        
        # Get the final file
        self.target_file_generation(self.base_saif_template, saif_dict, self.saif_result_file_path)
        
        