# kamal Al Molk Writing

**Table of contents:**

1. [Introduction](#introduction)

2. [Writing Folder](#writng-folder)
    1. [functions](#functions)
    2. [Outputs](#outputs)
    3. [Main Class](#main-class)

---

## Introduction

in this library, a code for Kamal Al Molk robot writing is written.
this is a brief explaination of code performance:
1. take a text and save to svg file
2. create a standard gcode for svg file
3. convert standard gcode to our local grammer 

---

### writng-folder
all of file is in this directory

---

### functions
we have three classes in this section:
- [create_SVG Class](#create_svg)
- [svg_to_gcode Class](#svg_to_gcode)
- [standard_to_nonstandard Class](#standard_to_nonstandard)

#### <span style="color:yellow"> create_svg </span>
in this class, the user give the text(persian,or english) and the text will be saved in an svg file named "output.svg",
the size of text,size of svg surface, and text position can be changed manually. 


#### <span style="color:yellow"> svg_to_gcode </span>
in this class, there is a code that creates a standard gcode from former svg file, in order to create a local compiler
with some parameters, that some of them are irrelevant to our robot,but their existance is necessary.

#### <span style="color:yellow"> standard_to_nonstandard </span>
in this class, the code convert standard gcode to local grammer for kamal, for this, we can map each standard gcode 
order to our case.
for example, "M5" is an equivalent for "c14" that means "pen up", or "M3" is an equivalent for "c13" that means
"pen down".

---

### Outputs
Each time that we run our program, we can see its outputs here.And that output contains an SVG file, and a TXT file,
that includes lines of G-Codes for hardware parts of project, and a TXT file for kamal grammer. 

---

### Main Class
in this class, we create an object from each class in "functions" directory, and run them sequentially in write() 
function.