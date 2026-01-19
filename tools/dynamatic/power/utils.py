import datetime
from string import Template

import os

# Reload string.Template method


class NewTemplate(Template):
    delimiter = '%'

################################################################
# Function Definition
################################################################


def get_date():
    """
        This function will get the running date of the script in isoformat
    """
    current_date = datetime.date.today()
    str_current_data = current_date.isoformat()

    return str_current_data


def remove_duplicate(file_path):
    """
        This function checks whether the given path exists or not
        if the file exist, delete it
    """
    if os.path.exists(file_path):
        print("[WARNING] " + file_path + " already exists, deleted!")
        os.remove(file_path)


def check_else_create(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def clean_all(folder_path):
    """
        This function will clean all test related files (log, different test folders)
    """
    # Check whether the overall folder exists or not
    if not os.path.exists(folder_path):
        print("[Cleaning] {} Cleaned".format(folder_path))
    else:
        # Folder exists, delete it
        command = "rm -r " + folder_path
        # Delete the folder
        os.system(command)
        print("[Cleaning] {} Cleaned".format(folder_path))


def target_file_generation(template_file, substitute_dict, target_path):
    """
        This file will generate the desired file based on the given template,
        replacement dictionary, and move the file to the desired locatino
    """
    # Write the new file
    with open(target_path, "w") as f:
        # Substitute the corresponding location in the template file
        s = NewTemplate(template_file)
        f.write(s.substitute(substitute_dict))
        f.close()
