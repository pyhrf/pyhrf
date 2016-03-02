from pyhrf.retreat import locator


def test_locate_harvard_oxford():
    # Left thalamus coordinates in MNI
    coords = [-6., -16., 9.]
    label = 'Left Thalamus'
    assert(locator.locate_harvard_oxford(coords,
                                         ho_atlas_name='sub-maxprob-thr25-1mm')
           == label)

    # Heschl's gyrus right coordinates in MNI
    coords = [43., -22., 9.]
    label = "Heschl's Gyrus (includes H1 and H2)"
    assert(locator.locate_harvard_oxford(coords, symmetric_split=False) ==
           label)
    assert(locator.locate_harvard_oxford(coords, symmetric_split=True) ==
           label + ', right part')

