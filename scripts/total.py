import json


section = 'WestDeckBox.nwd'
# branches = []

# recursively explore children is found
def getChildren(x):
	print (x.Parent.DisplayName)
	children = []
	for c in x.Children:
		children.append(getChildren(c))

	if len(children) > 0:
		return {x.DisplayName: children}
	else:
		return x.DisplayName


# select section
root = doc.Models[0].RootItem
for c in root.Children:
	if c.DisplayName == section:
		print(type(c.Parent.DisplayName))
		branches = getChildren(c)
		print(branches)
		
		with open(section+'_aggregation.json', 'w') as f:
			json.dump(branches, f)
	 
	 
# total = 0


# for c in root.Children:
# 	t = getTotal(c)
# 	total += t
# 	print c.DisplayName, " | ", t

# print "Total items: ", total


# selection = doc.CurrentSelection.SelectedItems

# if any(selection):
#     mi = selection[0]
#     for pc in mi.PropertyCategories:
#         print('\n')
#         print('Property Category: {} ({})'.format(pc.DisplayName, pc.Name))
#         for dp in pc.Properties:
#             if dp.Value.IsDisplayString:
#                 value = dp.Value.ToString()
#             elif dp.Value.IsDateTime:
#                 value = dp.Value.ToDateTime().ToShortTimeString()
#             else:
#                 value = dp.Value.ToString()

#             print('\t{} ({}): {}'.format(dp.DisplayName, dp.Name, value))