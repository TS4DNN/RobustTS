class ATSUtils(object):

    # 计算量大可约减
    # 如果这里选择全部c92计算量太复杂的话
    # 可以选择9个里最大的一个or两个，和剩下8个的组合
    def get_p_q_list(self, n, i):
        num_list = list(range(n))
        num_list.remove(i)
        import itertools
        pq_list = []
        # 抛掉一个点,剩下的排列组合选2个 C92
        for pq in itertools.combinations(num_list, 2):
            pq_list.append(pq)
        return pq_list
