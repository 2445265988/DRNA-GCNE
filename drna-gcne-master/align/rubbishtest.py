from input import read_dbp15k_input
file="../data/dbp15K/zh_en/mtranse/0_3/"
kg1, kg2, sup_ent1, sup_ent2, ref_ent1, ref_ent2, total_triples_num, total_ent_num, total_rel_num, rel_id_mapping=read_dbp15k_input(file)
for ent in sup_ent1:
    print(ent)