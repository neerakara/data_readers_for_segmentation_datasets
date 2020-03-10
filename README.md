# Data readers for segmentation datasets

Check 'main.py' to see how to read the datasets.
Set paths for saving pre-processed datasets in 'datapaths.py'

So far, this repository contains code for reading the following datasets:

Brain datasets:
<ul style="list-style-type:square;">
  <li>HCP (http://www.humanconnectomeproject.org/). For this dataset, there are T1w and T2w images for each subject. Segmentation labels have been generated via FreeSurfer. </li>
  <li>ABIDE (http://fcon_1000.projects.nitrc.org/indi/abide/). This dataset contains T1w images from several scanners. So far, we have pre-processed the data from two of these - Caltech and Stanford. Segmentation labels have been generated via FreeSurfer. </li>
</ul>

Cardiac datasets:
<ul style="list-style-type:square;">
  <li>ACDC (https://www.creatis.insa-lyon.fr/Challenge/acdc/). </li>
  <li>RVSC (http://rvsc.projets.litislab.fr/). </li>
</ul>

Prostate datasets:
<ul style="list-style-type:square;">
  <li>NCI (https://wiki.cancerimagingarchive.net/display/Public/NCI-ISBI+2013+Challenge+-+Automated+Segmentation+of+Prostate+Structures). </li>
  <li>PIRAD (Private dataset from University Hospital of Zurich). </li>
</ul>
