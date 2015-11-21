function actLabels = makeActLabels(activity_labels, actList, useIdle)

actLabels = activity_labels(actList);

if (useIdle)
   actLabels = ['Idle' ; actLabels];
end