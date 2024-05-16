# clf = SVC(gamma="auto")
# clf.fit(X_train, y_train)
#
# # Train another classifier with sample weighting
# weighted_clf = SVC(gamma="auto")
# weighted_clf.fit(X_train, y_train, sample_weight=gradient_descent(X_train, y_train))
#
# # Train another one with grid search sample weighting
# grid_weighted_clf = SVC(gamma="auto")
# grid_weighted_clf.fit(
#     X_train, y_train, sample_weight=grid_search_scratch(X_train, y_train)
# )
# # ---------------------------------------------------------------------------------------------
#
# # Show prediction vector without weighted training
# non_weighted_pred = clf.predict(X_test)
#
# # Show prediction vector with weighted training
# weighted_pred = weighted_clf.predict(X_test)
# grid_weighted_pred = grid_weighted_clf.predict(X_test)
#
# # Metrics for non-weighted samples model
# f1_no_weights = f1_score(y_test, non_weighted_pred)
# precision_no_weights = precision_score(y_test, non_weighted_pred)
# recall_no_weights = recall_score(y_test, non_weighted_pred)
# auc_roc_no_weights = roc_auc_score(y_test, non_weighted_pred)
#
# # Metrics for weighted samples model
# f1_with_weights = f1_score(y_test, weighted_pred)
# precision_with_weights = precision_score(y_test, weighted_pred)
# recall_with_weights = recall_score(y_test, weighted_pred)
# auc_roc_with_weights = roc_auc_score(y_test, weighted_pred)
#
# # Metrics for grid weighted samples model
# f1_with_grid_weights = f1_score(y_test, grid_weighted_pred)
# precision_with_grid_weights = precision_score(y_test, grid_weighted_pred)
# recall_with_grid_weights = recall_score(y_test, grid_weighted_pred)
# auc_roc_with_grid_weights = roc_auc_score(y_test, grid_weighted_pred)
#
# rounding = 3
#
# results = [
#     [
#         "Non-weighted",
#         round(f1_no_weights, rounding),
#         round(precision_no_weights, rounding),
#         round(recall_no_weights, rounding),
#         round(auc_roc_no_weights, rounding),
#     ],
#     [
#         "Weighted",
#         round(f1_with_weights, rounding),
#         round(precision_with_weights, rounding),
#         round(recall_with_weights, rounding),
#         round(auc_roc_with_weights, rounding),
#     ],
#     [
#         "Grid Weighted",
#         round(f1_with_grid_weights, rounding),
#         round(precision_with_grid_weights, rounding),
#         round(recall_with_grid_weights, rounding),
#         round(auc_roc_with_grid_weights, rounding),
#     ],
# ]
#
# headers = ["Model", "F1-score", "Precision", "Recall", "ROC-AUC"]
#
# print(tabulate(results, headers=headers))
#
# print()

# # Ploting input data vs. prediction
# # plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', marker='o', label='Test samples')
# fig, ax = plt.subplots(2, 2, figsize=(23, 14), width_ratios=[1, 1.25,1,1])
# # ax[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', marker='o', label='Test samples')
# ax[0].scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color='blue', label='Class 0')
# ax[0].scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='red', label='Class 1')
# ax[1].scatter(X_test[:, 0], X_test[:, 1], c=test_weights, cmap='inferno', marker='o', label='Test samples')
# fig.colorbar(ax[1].scatter(X_test[:, 0], X_test[:, 1], c=test_weights, cmap='inferno'), ax=ax[1], label="Weight")
#
# # ax[2].scatter(X_test[:, 0], X_test[:, 1], c=non_weighted_pred, cmap='coolwarm', marker='x', label='Unweighted prediction')
# ax[2].scatter(X_test[non_weighted_pred == 0][:, 0], X_test[non_weighted_pred == 0][:, 1], color='blue', marker='x', label='Class 0')
# ax[2].scatter(X_test[non_weighted_pred == 1][:, 0], X_test[non_weighted_pred == 1][:, 1], color='red', marker='o', label='Class 1')
#
# # ax[3].scatter(X_test[:, 0], X_test[:, 1], c=weighted_pred, cmap='coolwarm', marker='x', label='Weighted prediction')
# ax[3].scatter(X_test[weighted_pred == 0][:, 0], X_test[weighted_pred == 0][:, 1], color='blue', marker='x', label='Class 0')
# ax[3].scatter(X_test[weighted_pred == 1][:, 0], X_test[weighted_pred == 1][:, 1], color='red', marker='o', label='Class 1')
#
# ax[0].set_xlabel('Feature 1')
# ax[0].set_ylabel('Feature 2')
# ax[0].set_title('Unweighted data')
# ax[0].legend()
#
# ax[1].set_xlabel('Feature 1')
# ax[1].set_ylabel('Feature 2')
# ax[1].set_title('Data weights')
# ax[1].legend()
#
# ax[2].set_xlabel('Feature 1')
# ax[2].set_ylabel('Feature 2')
# ax[2].set_title('Unweighted prediction')
# ax[2].legend()
#
# ax[3].set_xlabel('Feature 1')
# ax[3].set_ylabel('Feature 2')
# ax[3].set_title('Weighted prediction')
# ax[3].legend()
# plt.tight_layout()
# plt.show()
#
# Ploting input data vs. prediction
# plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', marker='o', label='Test samples')
# fig, ax = plt.subplots(3, 2, figsize=(16, 14))
#
# # ax[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', marker='o', label='Test samples')
# ax[0][0].scatter(
#     X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color="blue", label="Class 0"
# )
# ax[0][0].scatter(
#     X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color="red", label="Class 1"
# )
# ax[0][1].scatter(
#     X_test[:, 0],
#     X_test[:, 1],
#     c=test_weights,
#     cmap="inferno",
#     marker="o",
# )
# fig.colorbar(
#     ax[0][1].scatter(X_test[:, 0], X_test[:, 1], c=test_weights, cmap="inferno"),
#     label="Weight",
# )
#
# # ax[2].scatter(X_test[:, 0], X_test[:, 1], c=non_weighted_pred, cmap='coolwarm', marker='x', label='Unweighted prediction')
# ax[1][0].scatter(
#     X_test[non_weighted_pred == 0][:, 0],
#     X_test[non_weighted_pred == 0][:, 1],
#     color="blue",
#     marker="x",
#     label="Class 0",
# )
# ax[1][0].scatter(
#     X_test[non_weighted_pred == 1][:, 0],
#     X_test[non_weighted_pred == 1][:, 1],
#     color="red",
#     marker="x",
#     label="Class 1",
# )
#
# # ax[3].scatter(X_test[:, 0], X_test[:, 1], c=weighted_pred, cmap='coolwarm', marker='x', label='Weighted prediction')
# ax[1][1].scatter(
#     X_test[weighted_pred == 0][:, 0],
#     X_test[weighted_pred == 0][:, 1],
#     color="blue",
#     marker="x",
#     label="Class 0",
# )
# ax[1][1].scatter(
#     X_test[weighted_pred == 1][:, 0],
#     X_test[weighted_pred == 1][:, 1],
#     color="red",
#     marker="x",
#     label="Class 1",
# )
#
# ax[2][0].scatter(
#     X_test[grid_weighted_pred == 0][:, 0],
#     X_test[grid_weighted_pred == 0][:, 1],
#     color="blue",
#     marker="x",
#     label="Class 0",
# )
# ax[2][0].scatter(
#     X_test[grid_weighted_pred == 1][:, 0],
#     X_test[grid_weighted_pred == 1][:, 1],
#     color="red",
#     marker="x",
#     label="Class 1",
# )
# ax[2][1].scatter(
#     X_test[:, 0],
#     X_test[:, 1],
#     c=grid_test_weights,
#     cmap="inferno",
#     marker="o",
# )
# fig.colorbar(
#     ax[2][1].scatter(X_test[:, 0], X_test[:, 1], c=grid_test_weights, cmap="inferno"),
#     label="Weight",
# )
#
# ax[0][0].set_xlabel("Feature 1")
# ax[0][0].set_ylabel("Feature 2")
# ax[0][0].set_title("Unweighted data")
# ax[0][0].legend()
#
# ax[0][1].set_xlabel("Feature 1")
# ax[0][1].set_ylabel("Feature 2")
# ax[0][1].set_title("Data weights")
# ax[0][1].legend()
#
# ax[1][0].set_xlabel("Feature 1")
# ax[1][0].set_ylabel("Feature 2")
# ax[1][0].set_title("Unweighted prediction")
# ax[1][0].legend()
#
# ax[1][1].set_xlabel("Feature 1")
# ax[1][1].set_ylabel("Feature 2")
# ax[1][1].set_title("Weighted prediction")
# ax[1][1].legend()
#
# ax[2][0].set_xlabel("Feature 1")
# ax[2][0].set_ylabel("Feature 2")
# ax[2][0].set_title("Grid Weighted prediction")
# ax[2][0].legend()
#
# ax[2][1].set_xlabel("Feature 1")
# ax[2][1].set_ylabel("Feature 2")
# ax[2][1].set_title("Grid search weights")
# ax[2][1].legend()
#
# plt.tight_layout()
# plt.show()
