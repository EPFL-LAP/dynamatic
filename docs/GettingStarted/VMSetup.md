# VM Setup Instructions

We provide a virtual machine (VM) which contains a pre-built/ready-to-use version of our entire toolchain except for the [simulator](../UserGuide/AdvancedBuild.md#6-modelsimquesta-installation) which the user must install and add to path manually. 

## Add Virtual Machine
DynamaticVM requires ~40GB of hard disk and 4GB of memory to dedicate to the VM.  
On Linux, you need to 
- install exFAT package to mount the USB drive.  
- CenOS/RedHat: sudo yum install exfat-utils fuse-exfat  
- Ubuntu/Debian: sudo apt-get install exfat-fuse exfat-utils.  

If you do not have the USB available, you can download the VM image [here](https://drive.google.com/file/d/1s86dzU8jbSSdh13DctS922OKoACgvVD5/view).

The Dynamatic virtual machine is compatible with Virtualbox 5.2 or later releases. You can use it to simply follow the tutorial (available in the [repository's documentation](Tutorials/Introduction/Introduction.md)) or as a starting point to use/modify Dynamatic in general.

> [!NOTE]
> Note that Dynamatic's repository on the VM does not track the `main` branch but a branch specifically made for the tutorial. If you would like to build Dynamatic's latest version from the VM, you can checkout the `main` branch and use the [regular build instructions](../GettingStarted/InstallDynamatic.md#build-instructions) from the top-level README to build the project's latest version.

To install the virual machine:
1. Extract the `.zip` in your local folder.
> The .vbox file contains all the settings required to run the VM, while the .vdi file
contains the virtual hard drive.  

2. Add DynamaticVM with the following steps:
- Click on `Machine`->`+Add`, and select the file `DynamaticVM.vbox`.
<img src="">  
- Start DynamaticVM by clicking on `Start`

## Troubleshooting
In case of resolution issues, go to “View”->”+Virtual Screen 1” and select a zoom factor that best suit your screen, for
example 300. Then, go to `View` and select the option `Full screen mode`.

## Inside the VM

If everything went well, after launching the image you should see Ubuntu's splash screen and be dropped into the desktop directly. Below are some important things about the guest OS running on the VM.

- The VM runs Ubuntu 22.04 LTS. Any kind of "system/program error" reported by Ubuntu can safely be dismissed or ignored.
- The user on the VM is called *dynamatic*. **The password is also dynamatic**.
- On the left bar you have icons corresponding to a file explorer, a terminal, a web browser (Firefox), and an IDE (VSCode) which opens Dynamatic by default.
  - There are a couple default Ubuntu settings you may want to modify for your convenience. You can open Ubuntu settings by clicking the three icons at the top right of the Ubuntu desktop and then selecting **Settings**.
  - You can change the default display resolution (1920x1080) by clicking on the **Displays** tab on the left, then selecting another resolution in the **Resolution** dropdown menu.
  - You can change the default keyboard layout (English US) by clicking on the **Keyboard** tab on the left. Next, click on the + button under **Input Sources**, then, in the pop-menu that appears, click on the three vertical dots icon, scroll down the list, and click **Other**. Find your keyboard layout in the list and double-click it to add it to the list of input sources. Finally, drag your newly added keyboard layout above **English (US)** to start using it.
- When running commands for Dynamatic from the terminal, make sure you first `cd` to the `dynamatic` subfolder.
  - Since the user is also called *dynamatic*, `pwd` should display `/home/dynamatic/dynamatic` when you are in the correct folder.
  - You can run `./update-dynamatic.sh` from the dynamatic subfolder to pull latest changes from the repository and automatically rebuild the project.
