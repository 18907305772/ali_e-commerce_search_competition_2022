import random


class Dataloader1(object):
    def __init__(self, list1: list, batch_size: int):
        self.dataset = list1
        self.batch_size = batch_size
        self.maximum_steps = int(len(self.dataset) / batch_size)
        self.steps_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.steps_count > self.maximum_steps:
            self.steps_count = 0
            raise StopIteration
        self.steps_count += 1
        batch = random.sample(self.dataset, self.batch_size)
        batch_list1 = []
        for l1 in batch:
            batch_list1.append(l1)
        return batch_list1


class Dataloader2(object):
    def __init__(self, list1: list, list2: list, batch_size: int):
        self.dataset = [(l1, l2) for l1, l2 in zip(list1, list2)]
        self.batch_size = batch_size
        self.maximum_steps = int(len(self.dataset) / batch_size)
        self.steps_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.steps_count > self.maximum_steps:
            self.steps_count = 0
            raise StopIteration
        self.steps_count += 1
        batch = random.sample(self.dataset, self.batch_size)
        batch_list1 = []
        batch_list2 = []
        for l1, l2 in batch:
            batch_list1.append(l1)
            batch_list2.append(l2)
        return batch_list1, batch_list2


class Dataloader3(object):
    def __init__(self, list1: list, list2: list, list3: list, batch_size: int):
        self.dataset = [(l1, l2, l3) for l1, l2, l3 in zip(list1, list2, list3)]
        self.batch_size = batch_size
        self.maximum_steps = int(len(self.dataset) / batch_size)
        self.steps_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.steps_count > self.maximum_steps:
            self.steps_count = 0
            raise StopIteration
        self.steps_count += 1
        batch = random.sample(self.dataset, self.batch_size)
        batch_list1 = []
        batch_list2 = []
        batch_list3 = []
        for l1, l2, l3 in batch:
            batch_list1.append(l1)
            batch_list2.append(l2)
            batch_list3.append(l3)
        return batch_list1, batch_list2, batch_list3


class Dataloader4(object):  # 修改batch采样方式
    def __init__(self, list1: list, list2: list, list3: list, list4: list, batch_size: int):
        self.dataset = [(l1, l2, l3, l4) for l1, l2, l3, l4 in zip(list1, list2, list3, list4)]
        self.batch_size = batch_size
        self.maximum_steps = int(len(self.dataset) / batch_size)
        self.steps_count = 0
        random.shuffle(self.dataset)

    def __iter__(self):
        return self

    def __next__(self):
        if self.steps_count >= self.maximum_steps:
            self.steps_count = 0
            raise StopIteration
        batch = self.dataset[self.steps_count * self.batch_size: (self.steps_count + 1) * self.batch_size]
        self.steps_count += 1
        # batch = random.sample(self.dataset, self.batch_size)
        batch_list1 = []
        batch_list2 = []
        batch_list3 = []
        batch_list4 = []
        for l1, l2, l3, l4 in batch:
            batch_list1.append(l1)
            batch_list2.append(l2)
            batch_list3.append(l3)
            batch_list4.append(l4)
        return batch_list1, batch_list2, batch_list3, batch_list4


class Dataloader5(object):  # 修改batch采样方式
    def __init__(self, list1: list, list2: list, list3: list, list4: list, list5: list, batch_size: int):
        self.dataset = [(l1, l2, l3, l4, l5) for l1, l2, l3, l4, l5 in zip(list1, list2, list3, list4, list5)]
        self.batch_size = batch_size
        self.maximum_steps = int(len(self.dataset) / batch_size)
        self.steps_count = 0
        random.shuffle(self.dataset)

    def __iter__(self):
        return self

    def __next__(self):
        if self.steps_count >= self.maximum_steps:
            self.steps_count = 0
            raise StopIteration
        batch = self.dataset[self.steps_count * self.batch_size: (self.steps_count + 1) * self.batch_size]
        self.steps_count += 1
        # batch = random.sample(self.dataset, self.batch_size)
        batch_list1 = []
        batch_list2 = []
        batch_list3 = []
        batch_list4 = []
        batch_list5 = []
        for l1, l2, l3, l4, l5 in batch:
            batch_list1.append(l1)
            batch_list2.append(l2)
            batch_list3.append(l3)
            batch_list4.append(l4)
            batch_list5.append(l5)
        return batch_list1, batch_list2, batch_list3, batch_list4, batch_list5
