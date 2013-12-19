.. _paradigm:

Paradigm CSV format
*******************

To encode the experimental paradigm, the CSV file format comprises 5 columns:

    1. Session number (integer)
    2. Event type, as string with double quotes
    3. Onset (float in sec.)
    4. Duration (float sec.). 0 for event-related paradigm
    5. Amplitude (float, default is 1.).


Example (extracted from pyhrf/python/datafiles/paradigm_loc.csv):

.. csv-table:: Localizer paradigm
   :header: "Session", "Experimental Condition", "Onset", "Duration", "Amplitude"
   :widths: 10, 10, 10, 10, 10
   :delim: space
   :file: ./paradigm_sample.csv


