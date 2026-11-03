# Historical City Maps Semantic Segmentation Dataset

This dataset includes a total of 635 annotated image patches from historical city maps. It is designed for the semantic segmentation of the maps into 3 (map content, road network, background frame) or 5 semantic classes (building blocks, non-built, water, road network, background frame). 330 patches are taken from maps of the city of Paris, while the 305 others are taken from a balanced corpus of city maps from all around the world. 

## About Data collection methodology

### Paris dataset

Two third of the data patches comes from the collections of the Bibliothèque nationale de France (French national library, BnF), while the other third comes from the collections of the Bibliothèque historique de la Ville de Paris (Historical library of the City of Paris, BHVP). 90% of the original maps have a scale comprised between 1:25'000 and 1:2'000. The BnF maps cover a period from 1760 to 1949, while the BHVP maps cover a period from 1787 to 1994. However, most maps were published between 1800 and 1950. Three hundred and thirty 1000x1000 pixel map patches were randomly cut from the more than 1500 maps included in the original corpus, and manually annotated for semantic segmentation. More details can be found in the BnF metadata. For the BHVP, most maps were published by the Parisian administration, namely the Prefecture de la Seine, for urban planning purposes. The most common scale is 1:5'000.

#### Paris dataset metadata
Metadata include:
* ark identifier,
* patch name,
* leaflet number,
* language,
* map scale,
* publisher,
* publisher city,
* title,
* year of publication,
* image weblink,
* metadata weblink,
* subject,
* source location,
* physical format.

### World dataset

The world corpus was established using cartographic reference manuals and digitized collections available online. The period of publication of these maps ranges from 1700 to 1950. During the constitution of the corpus, a balance was made between Western and non-Western maps. In total, the corpus includes 256 maps from 32 heritage institutions, the largest of which are the BnF (59 maps), the Library of Congress (31), the Harvard Library (28), the David Rumsey collection (22), and the University Bordeaux-Montaigne (19). The 256 maps represent 182 different cities in 90 countries around the world. The regions with the most maps are Eastern Europe & Central Asia (34), Western Europe (34), and East Asia (30). Conversely, the regions accounting for the fewest maps are Oceania (9), Subsaharian Africa (15), and Middle East (17). The urban form of each map in the World corpus was manually classified into three categories: regular (95 maps), irregular (68), or mixed (93). More details can be found in the associated metadata file. Training patches were randomly cut from each of the 256 maps, and then 49 additional patches were randomly cut to constitute the val/test set.

#### World dataset metadata
Metadata include:
* patch name,
* weblink,
* source collection,
* year of publication,
* city represented,
* country,
* region,
* urban shape of the city represented,
* country of publication,
* title.

### Annotation ontology

#### 2+1 classes ontology

* Frame (black), i.e. the non-mapping part of the document. It contains the frame as well as the illustrations, illuminations, surrounding texts and medallions;
* Road network (white). This class includes roads and streets as well as railroads and bridges;
* Map content (magenta), i.e. all the cartographic content of the map, which does not belong to the road network.

#### 4+1 classes ontology
The frame and the road network classes are unchanged.
* Blocks (magenta), city blocks, detached houses, walls or anything similar to buildings;
* Water (blue), including rivers, canals, bodies of water, reservoirs, lakes and the sea;
* Non-built (cyan), which in fact includes all non-aquatic unbuilt land, including wasteland, meadows, crops, forests, but also urban unbuilt land, such as parks, enclosed squares, inner courtyards.


### Organization of the data

Here you can descibe how the data is organized in this whole dataset. How the data is stored in all the files. You also have to brief about the naming convention of the files in different directories. 

```
dataset/
  -README.md
  -License - CC BY-NC 4.0
  -paris/
    -eval/
      -images/      (30 png images)
      -labels/      (30 png images)
    -images/        (300 png images)
    -labels/        (300 png images)
  -world/
    -eval/
      -images/      (49 png images)
      -labels/      (49 png images)
    -images/        (256 png images)
    -labels/        (256 png images)
  -classes.json
  -metadata_BnF_paris.xlsx
  -metadata_world.xlsx
```

## Online Repository link

* [City Maps Semantic Segmentation Dataset on Zenodo](https://zenodo.org/record/5513639) - Link to the data repository.

## Author

**Remi PETITPIERRE** - [remi.petitpierre@epfl.ch](mailto:remi.petitpierre@epfl.ch) - [ORCID](https://orcid.org/0000-0001-9138-6727) - [Github](https://github.com/RPetitpierre/) -  [Scholar](https://scholar.google.com/citations?user=4aMZmocAAAAJ&hl=en) - [ResearchGate](https://www.researchgate.net/profile/Remi-Petitpierre) 

## Use and Citation

For any mention of this dataset, please cite :

```
@misc{petitpierre_historical_2021,
  author    = {Petitpierre, R{\'{e}}mi},
  title     = {Historical City Maps Semantic Segmentation Dataset},
  year      = {2021},
  howpublished = {\url{https://zenodo.org/record/5513639}},
  doi       = {10.5281/zenodo.5513639}
}
```
```
@inproceedings{petitpierre_generic_2021,
  author    = {Petitpierre, R{\'{e}}mi and Kaplan, Fr{\'{e}}d{\'{e}}ric and di Lenardo, Isabella},
  title     = {Generic Semantic Segmentation of Historical Maps},
  booktitle = {Proceedings of the Workshop on Computational Humanities Research (CHR 2021)},
  year      = {2021},
  eventdate = {2021-11-17/2021-11-19},
  publisher = {CEUR},
  address   = {Amsterdam, NL}
}
```

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License](https://creativecommons.org/licenses/by-nc/4.0/deed.en).

## Liability

We do not assume any liability for the use of this dataset.

