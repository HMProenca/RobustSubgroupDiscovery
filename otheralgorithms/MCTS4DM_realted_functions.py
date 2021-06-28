def run_MCTS4DM_wrapper(algorithmname, beam_width, number_rules_SSD, datasetname, df, task, depthmax,attribute_names,number_targets):
    if algorithmname == "seq-cover":
        conf_file = read_csvfile('./otheralgorithms/DSSD/bin/tmp_sequential.conf')
    elif algorithmname == "top-k":
        conf_file = read_csvfile('./otheralgorithms/DSSD/bin/tmp_topk.conf')
        conf_file[10] = ['topK = ' + str(int(number_rules_SSD.loc[datasetname, "number_rules"]))]
        nrows = df.shape[0]
        if nrows < 2000 and task == "single-nominal":
            conf_file[14] = ['searchType = ' + "dfs"]
        else:
            conf_file[14] = ['searchType = ' + "beam"]
    else:
        raise Exception("Wrong aglorithm name")

    conf_file[19] = ['beamWidth = ' + str(int(beam_width))]
    conf_file[15] = ['maxDepth = ' + str(min(int(depthmax), 10))]

    if task == "multi-nominal" or task == "single-nominal":
        conf_file[23] = ['measure = WKL']
        # conf_file[24] = ['WRAccMode = 1vsAll']
    elif task == "multi-numeric" or task == "single-numeric":
        conf_file[23] = ['measure = meantest']
        conf_file[24] = ['WRAccMode = 1vsAll']
    else:
        raise Exception("Wrong task name")

    write_file_dssd(conf_file, './otheralgorithms/DSSD/bin/tmp.conf')

    # check if path exists
    if not os.path.exists('.//otheralgorithms//DSSD//xps//dssd'):
        os.makedirs('.//otheralgorithms//DSSD//xps//dssd')
    else:
        shutil.rmtree('.//otheralgorithms//DSSD//xps//dssd')
        os.makedirs('.//otheralgorithms//DSSD//xps//dssd')

    # change target variable file - target variables are at the end!
    name_targets = attribute_names[-number_targets:]
    targets_file = pd.read_csv('./otheralgorithms/DSSD/data/datasets/tmp/emmModel.emm', delimiter="=", header=None)
    targets_file.iloc[1, 1] = ' ' + ','.join([tg_name for tg_name in name_targets])
    targets_file.to_csv('./otheralgorithms/DSSD/data/datasets/tmp/tmp.emm', index=False, sep="=", header=False)

    # run DSSD
    timespent = time()
    os.chdir("./otheralgorithms/DSSD/bin")
    call(["emc64-mt-modified.exe"])
    # call(["dssd64.exe"])
    os.chdir("../../")
    timespent = time() - timespent
    os.remove("./otheralgorithms/DSSD/data/datasets/tmp/tmp.arff")

    # read output files
    auxfiles = [path for path in os.listdir('./otheralgorithms/DSSD/xps/dssd/')]
    generated_xp = './otheralgorithms/DSSD/xps/dssd/' + auxfiles[-1]  # last one
    timestamp = generated_xp.split('-')[1]
    # find transaction ids of subgroups
    generated_xp_subsets_path = generated_xp + '/subsets'
    all_generated_subgroups_files = [generated_xp_subsets_path + '/' + x
                                     for x in os.listdir(generated_xp_subsets_path)]
    # find descriptions of subgroups
    if algorithmname == "top-k":
        description_files = generated_xp + '/' + "stats1-" + timestamp + ".csv"
    elif algorithmname == "seq-cover":
        description_files = generated_xp + '/' + "stats2-" + timestamp + ".csv"

    # count number of items per subgroup
    descriptions = read_csvfile(description_files)
    # columnames, typevar, limits = info4prediction(df.iloc[:, :-number_targets], number_targets)
    # patterns4prediction = make_patterns4prediction(descriptions, columnames, typevar, limits)
    # Test dataset
    # nrows_test = Y_test.shape[0]
    # bitsets_subgroups = findbitsets(patterns4prediction,X_test,Y_test)

    nitems = []
    for row in descriptions[1:]:
        # count items
        nitems.append(1 + row[0].count("&&"))

    subgroup_sets_support = []
    subgroup_sets_support_bitset = []
    support_union = set()
    nb_subgroups = 0
    rules_supp = []
    for subgroup_file in all_generated_subgroups_files:
        aux_subgroup = read_csvfile(subgroup_file)[2:]
        subgroup_biset = [row[0] for row in aux_subgroup]
        subgroup_index = set(i for i, x in enumerate(subgroup_biset) if x == '1')
        subgroup_sets_support.append(subgroup_index)
        subgroup_sets_support_bitset.append(indexes2bitset(subgroup_index))
        support = len(subgroup_index)
        rules_supp.append(support)
        nb_subgroups += 1

    return nitems, subgroup_sets_support_bitset, timespent



algorithmname = "MCTS4DM"
savefile = algorithmname + "a_summary.txt"
top_k = 10  # number of patterns to keep
offset = 0
# First_Launching=args.First_Launching
timebudget = 3600
depthmax = 5.0
#

# does not run "sonar", "german", "adult"
datasetnames = ["sonar", "haberman", "breastCancer", "australian", "magic", "iris",
                "balance", "CMC", "page-blocks", "glass", "dermatology", "kr-vs-k"]
datasetnames = ["magic"]
print(
    "dataset,kl_supp,avg_supp,wkl_supp,kl_usg,avg_usg,wkl_usg,wacc_supp,wacc_usg,kl_union,wkl_union,wacc_union,union_supp,wkl_total,jacc_avg,nr,avg_items,timespent",
    file=open(savefile, "w"))
for datasetname in datasetnames:
    top_k = lenSDD[datasetname]
    file = "C:/Users/hugoadmin/Desktop/MDLsubgroupdiscovery/FSSD/PreparedDatasets/" + datasetname + ".csv"

    attributes, types = transform_dataset_to_attributes(file, class_attribute, delimiter=delimiter)
    full_attributes = attributes[:]
    dataset, header = readCSVwithHeader(file, numberHeader=[a for a, t in zip(attributes, types) if t == 'numeric'],
                                        delimiter=delimiter)
    attributes = attributes[offset:offset + nb_attributes]
    types = types[offset:offset + nb_attributes]
    wanted_label = dataset[0]["class"]

    new_dataset, positive_extent, negative_extent, alpha_ratio_class, _ = transform_dataset(dataset, attributes,
                                                                                            class_attribute,
                                                                                            wanted_label)
    #    for row in new_dataset:
    #    	row['class']=row['positive']
    #    	del row['positive']
    dfaux = pd.read_csv(file, sep="\t")
    if types[0] == 'simple':
        for col in dfaux:
            dfaux[col] = dfaux[col].astype('category')
            dfaux[col] = dfaux[col].cat.codes
    elif types[0] == 'numeric':
        dfaux["class"] = dfaux["class"].astype('category')
        dfaux["class"] = dfaux["class"].cat.codes
    dataset = dfaux.to_dict(orient="records")
    new_dataset = deepcopy(dataset)

    writeCSVwithHeader(new_dataset, './MCTS4DM/datasets/tmp/properties.csv', selectedHeader=attributes, delimiter='\t',
                       flagWriteHeader=True)
    writeCSVwithHeader(new_dataset, './MCTS4DM/datasets/tmp/qualities.csv', selectedHeader=['class'], delimiter='\t',
                       flagWriteHeader=True)
    find_conf = read_file_conf('./MCTS4DM/tmpModel.conf')
    # print(find_conf)
    if types[0] == 'simple':
        find_conf[2] = ['attrType = Nominal']
    elif types[0] == 'numeric':
        find_conf[2] = ['attrType = Numeric']
    find_conf[6] = ['maxOutput = ' + str(int(top_k))]
    find_conf[5] = ['nbIter = 50000']

    find_conf = [{'###CONF FILE FOR MCTS###': x[0]} for x in find_conf]
    # print(find_conf)
    writeCSVwithHeader(find_conf, './MCTS4DM/tmp.conf', selectedHeader=['###CONF FILE FOR MCTS###'], delimiter='\t',
                       flagWriteHeader=True)
    # for row in new_dataset:

    # 	print (row)

    if os.path.exists('.//MCTS4DM//results//tmp'):
        shutil.rmtree('.//MCTS4DM//results//tmp')
    os.chdir("./MCTS4DM")
    timespent = time()
    call(["java", "-jar", "MCTS4DM.jar", "tmp.conf"])
    timespent = time() - timespent
    os.chdir("../")

    generated_xp = './/MCTS4DM//results//tmp//' + os.listdir('.//MCTS4DM//results//tmp')[0] + '//support.log'
    # print(generated_xp)
    d = readCSV(generated_xp)
    subgroup_sets = []
    support_union = set()
    nb_patterns = 0

    rules_supp = []
    rules_usg = []

    nitems = []
    for subgroup_l in d:
        subgroup_sets.append(set([int(x) - 1 for x in subgroup_l[0].split(' ')]))
        #        rules_suppaux = [0,0]
        #        rules_suppaux[0] = len(subgroup_sets[-1]& positive_extent)
        #        rules_suppaux[1] = len(subgroup_sets[-1]& negative_extent)
        #        rules_supp.append(rules_suppaux)
        #        rules_usgaux = [0,0]
        #        rules_usgaux[0] = len(subgroup_sets[-1].difference(support_union)& positive_extent)
        #        rules_usgaux[1] = len(subgroup_sets[-1].difference(support_union)& negative_extent)
        #        rules_usg.append(rules_usgaux)
        support_union |= subgroup_sets[-1]
        nb_patterns += 1
        #        tpr_support_union=float(len(support_union & positive_extent))/float(len(positive_extent))
        #        fpr_support_union=float(len(support_union & negative_extent))/float(len(negative_extent))

    c_values = list(set([row["class"] for row in dataset]))
    count_cl = [0 for c in c_values]
    for row in dataset:
        for ic, c in enumerate(c_values):
            if row["class"] == c:
                count_cl[ic] += 1

    rules_usg = []
    for r in range(len(subgroup_sets)):
        previous_sets = [subgroup_sets[ii] for ii in range(r)]
        auxset = subgroup_sets[r].difference(*previous_sets)
        aux_usg = [0 for c in c_values]
        for idx in auxset:
            for ic, c in enumerate(c_values):
                # NOTE : dataset is used instead of newdataset because new has 2 more lines
                if dataset[idx]["class"] == c:
                    aux_usg[ic] += 1
        rules_usg.append(aux_usg)
    rules_usg = [rlusg for rlusg in rules_usg if sum(rlusg) != 0]

    rules_supp = []
    for r in range(len(subgroup_sets)):
        auxset = subgroup_sets[r]
        auxrule_supp = [0 for c in c_values]
        for idx in auxset:
            for ic, c in enumerate(c_values):
                # NOTE : dataset is used instead of newdataset because new has 2 more lines
                if dataset[idx]["class"] == c:
                    auxrule_supp[ic] += 1
        rules_supp.append(auxrule_supp)

    union_pattern = set().union(*subgroup_sets)
    supp_union = [0 for c in c_values]
    for idx in union_pattern:
        for ic, c in enumerate(c_values):
            if dataset[idx]["class"] == c:
                supp_union[ic] += 1
    union_supp = sum(supp_union)

    items_xp = './/MCTS4DM//results//tmp//' + os.listdir('.//MCTS4DM//results//tmp')[0] + '//result.log'
    f = open(items_xp, "r")
    aaa = f.readlines()
    aaa.pop(0)
    nitems = []
    for line in aaa:
        nitems.append(line.count(",") + 1)
    f.close()

    nr = nb_patterns
    union_supp = sum(supp_union)
    kl_supp, kl_usg, wkl_supp, wkl_usg, wkl_total, wacc_supp, wacc_usg, kl_union, wkl_union, wacc_union = \
        discoverymetrics(c_values, nr, rules_supp, rules_usg, count_cl, supp_union)

    nr = len(rules_supp)
    intersect = np.zeros([nr, nr], dtype=np.uint)
    jaccard = np.zeros([nr, nr])
    for kk in range(nr):
        intersect[kk, kk] = len(subgroup_sets[kk])
        for kk2 in range(kk + 1, nr):
            intersect[kk, kk2] = len(subgroup_sets[kk] & subgroup_sets[kk2])

    for rr in itertools.combinations(range(nr), 2):
        inter = intersect[rr]
        supp1 = intersect[(rr[0], rr[0])]
        supp2 = intersect[(rr[1], rr[1])]
        jaccard[rr] = inter / (supp1 + supp2 - inter)

        # average over all possible cases
    uptm = np.triu_indices(nr, 1)
    jacc_avg = np.sum(jaccard) / len(uptm[0])
    jacc_consecutive_avg = np.mean(np.diagonal(jaccard, 1))
    avg_supp = np.mean([sum(rules_supp[r]) for r in range(nr)])
    avg_usg = np.mean([sum(rusg) for rusg in rules_usg])
    savefile = algorithmname + "a_summary.txt"
    print(
        "%s , %.4f , %.4f , %.4f , %.4f , %.4f , %.4f , %.4f , %.4f  , %.4f , %.4f , %.4f  , %.4f , %.4f , %.4f , %.4f , %.4f ,  %.4f" \
        % (datasetname,
           kl_supp, \
           avg_supp, \
           wkl_supp, \
           kl_usg, \
           avg_usg, \
           wkl_usg, \
           wacc_supp, \
           wacc_usg, \
           kl_union, \
           wkl_union, \
           wacc_union, \
           union_supp, \
           wkl_total, \
           jacc_avg, \
           nr, \
           sum(nitems) / nr, \
           timespent), \
        file=open(savefile, "a"))
