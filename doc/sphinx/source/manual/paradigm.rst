.. _paradigm:

=========================
File format specification
=========================


Paradigm CSV format
*******************

To encode the experimental paradigm, the CSV file format comprises 5 columns:

1. Session number (integer)
2. Experimental Condition, as string with double quotes
3. Onset (float in s.)
4. Duration (float s.). 0 for event-related paradigm
5. Amplitude (float, default is 1.).


Example extracted from ``pyhrf/python/datafiles/paradigm_loc.csv``:

.. csv-table:: Localizer paradigm
   :header: "Session", "Experimental Condition", "Onset", "Duration", "Amplitude"
   :widths: 10, 10, 10, 10, 10
   :delim: space
   :file: ./paradigm_sample.csv

**NB**: Be careful not to end float onset and duration with a point ``.``
because it could result in python not being able to split the paradigm
correctly (end with a zero if necessary)

Contrast JSON format
********************

To encode contrasts definition, use the following **json format**:

.. code:: json

    {
        "contrast_name": "linear combination of conditions",
        "Audio-Video": "clicDaudio-clicDvideo"
    }

The names in the contrast definition **must** correspond to
the experimental conditions (**case sensitive**) defined in the above CSV file.
