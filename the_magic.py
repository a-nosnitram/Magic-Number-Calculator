
# THE MOST IMPORTANT PART
# calculation based on the ratio of the lengths of the index and ring fingers
# taken from https://pubmed.ncbi.nlm.nih.gov/21725330/

def the_calculation(ratio):
    return (20.577 - 9.201 * ratio)


# assumption: er = 1.4 * fl (iykyk)
def the_more_forgiving_calculation(ratio):
    return (20.577 - 9.201 * ratio) * 1.4
