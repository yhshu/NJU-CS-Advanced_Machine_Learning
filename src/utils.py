def dummy_data_collector(input_ids, labels=None):
    if input_ids is not None and labels is not None:
        assert len(input_ids) == len(labels)
    res = []
    for i in range(0, len(input_ids)):
        sample_dict = {}
        if input_ids is not None:
            sample_dict['input_ids'] = input_ids[i]
        if labels is not None:
            sample_dict['labels'] = labels[i]
        res.append(sample_dict)
    return res
