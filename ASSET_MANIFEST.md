# Asset manifest

Audit date: 2026-09-23. SHA-256 hashes were calculated from the original read-only files. Paths are local provenance, not destinations for Git. Dataset image bytes were not traversed or hashed en masse.

**Runtime assets loaded by the original backend:**

| Role | Original path | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| YOLO26s | `E:\SIH\SIH_Results\yolo26s_generic_baseline\weights\best.pt` | 20303429 | `e069ee7b5180bf35e7f71f908f47160ab9e70266d611374dd77e0a34e741eb8e` |
| ConvVAE | `E:\SIH\SIH_Results\vae_normal_seabed\checkpoints\best.pt` | 218365942 | `b950f60ce1358b72a9b2c58f1fd0ec955b8e7f8850205662d976356f87e043c9` |
| RealNVP | `E:\SIH\SIH_Results\latent_normalizing_flow\best_flow.pt` | 3714807 | `dbfc83c3baf1ac3a840b708b2b02b22c6a0753f1c969d7b18d623efc2e40ac65` |
| Latent mean | `E:\SIH\SIH_Results\latent_normalizing_flow\latent_mean.pt` | 2117 | `56771caed4bbecc6ff019d4f63a3e16664ade7f196462c8ef223e031da8f351c` |
| Latent std | `E:\SIH\SIH_Results\latent_normalizing_flow\latent_std.pt` | 2110 | `087742e906b695d06fa0a78fe7ddd31e3a02259d1569669fda0483273dbf3326` |

The original API percentile constants, now pinned in `config/live_api_v1.json` and used by `backend/saad_inference/evidence/live_api_v1.py`, match rounded values in `SIH_Results/saad_evidence_engine_v3/normalization_parameters.csv` (59 validation normal images). The live priority and uncertainty formulas differ from that offline v3 configuration. The checkpoints above are the files referenced by the loader; similarly named `last.pt`, v2 SSIM, MAR, and base YOLO weights are research assets.

## Dataset structure and integrity

| Prepared set | Split | Image files | Label files |
| --- | --- | ---: | ---: |
| SAAD_baseline | train | 6571 | 6616 |
| SAAD_baseline | val | 1097 | 1097 |
| SAAD_baseline | test | 544 | 544 |
| SAAD_VAE | train | 11475 | — |
| SAAD_VAE | val | 1403 | — |
| SAAD_VAE | test | 1408 | — |

Prepared detector manifest class/domain counts (manifest rows):

| Domain | Split | Anthropogenic | Normal |
| --- | --- | ---: | ---: |
| AI4Shipwrecks | train | 73 | 48 |
| AI4Shipwrecks | val | 14 | 6 |
| AI4Shipwrecks | test | 74 | 46 |
| GhostVision | train | 4291 | 1430 |
| GhostVision | val | 502 | 53 |
| GhostVision | test | 334 | 64 |
| Marine-PULSE | train | 0 | 88 |
| Marine-PULSE | val | 0 | 0 |
| Marine-PULSE | test | 0 | 0 |
| SubPipeMini2 | train | 686 | 0 |
| SubPipeMini2 | val | 522 | 0 |
| SubPipeMini2 | test | 26 | 0 |

Existing integrity evidence: `Datasets/audit/annotation_errors.csv` records missing_annotation=685. `Datasets/SAAD_baseline/integrity_subpipe_temporal_leakage.txt` records neighboring SubPipeMini2 frames crossing train/val and train/test splits (delta 1.0). This limits independence claims for that domain.

Three representative test images were visually inspected without altering them: a 1728×2476 AI4Shipwrecks side-scan frame with an empty YOLO label file, a 640×640 GhostVision sonar frame with two class-0 boxes, and a 2500×500 SubPipeMini2 strip with one class-0 box. Their exact filenames, image hashes, and inference outputs are in [BASELINE_VERIFICATION.md](BASELINE_VERIFICATION.md). Their different dimensions and annotation patterns make them useful fixed regression cases; their appearance alone does not establish model correctness.

Source dataset roots present: `AI4Shipwrecks`, `GhostVision_fixed`, `Marine_PULSE`, `SeabedObjects-Ship-and-Airplane-dataset-master`, `sss-crab-pot-detection-ds`, and `SubPipeMini2`. These were inspected structurally; no recursive image-byte reading was performed.

## Hashed artifact and result ledger

Every listed original file is an **EXTERNAL ASSET** for the clean repository. Calibration/result tables and reports remain research provenance. No model or result is copied into Git.

| Path | Bytes | SHA-256 |
| --- | ---: | --- |
| `E:\SIH\Datasets\audit\annotation_errors.csv` | 115761 | `077643ec74a68b9695fcae62f2bb5f6228199aaf8655fb7a5e7867ffd13254ee` |
| `E:\SIH\Datasets\audit\dataset_summary.csv` | 537 | `b98942385be3f85370eb56c319c65b640405c54c9b2a247affc860790f204c8c` |
| `E:\SIH\Datasets\audit\image_records.csv` | 3328011 | `20c7c1236263e008283d1224199cb0c098c1dee6d9d5057b2d9262e938b7f056` |
| `E:\SIH\Datasets\SAAD_baseline\data.yaml` | 124 | `cd1e08fe98969f9955ccc23a046de33dcfd68dcd659e6755d9af6cec1e43b90f` |
| `E:\SIH\Datasets\SAAD_baseline\dataset_summary.csv` | 567 | `0e112e92c8f89ed05195b895d3d9f2ff978ede1a49bbc6d3b8d5bb4390bed71b` |
| `E:\SIH\Datasets\SAAD_baseline\integrity_subpipe_temporal_leakage.txt` | 525 | `76de670ebc28fbdd4deaa4fcec1a1d383989f4a8f634fb749ffe7c6e0c5908d1` |
| `E:\SIH\Datasets\SAAD_baseline\manifest.csv` | 3610565 | `e95c7fea228c6e146ef943155c3bc8a81b060a4ea337c6c1880465f96bcd9a1a` |
| `E:\SIH\Datasets\SAAD_VAE\manifest.csv` | 5719917 | `eb206e32d9227f7b5ee6af588887ed76d62cc5757e40e0706a5344ec7a65ccb2` |
| `E:\SIH\SIH_Results\active_learning_retrain_250\acquisition\acquisition_pool.csv` | 616913 | `8bffb221cf5ba291d106d07284d82140b7b2652f096f22b45757a6b764a67135` |
| `E:\SIH\SIH_Results\active_learning_retrain_250\acquisition\initial_seed.csv` | 267024 | `4fcb7858bd4521ee9d439bb82642db71f515aa7d4aab06da2887e79410b14f8d` |
| `E:\SIH\SIH_Results\active_learning_saad\acquisition_pool.csv` | 3008965 | `579689f0c5b92e69c466a9af1e2709521714b73650f10a6ecb173694969a9a46` |
| `E:\SIH\SIH_Results\active_learning_saad\ACTIVE_LEARNING_SUMMARY.txt` | 707 | `0b156249292299100489aad3fe3c08296ab53253aa57b5502794256e58d8df5a` |
| `E:\SIH\SIH_Results\active_learning_saad\class_selection_summary.csv` | 2825 | `f98ec450d1e60436fe5cc24bbdec957de0e7dfe7b5ae674164adbada1f1ed9ba` |
| `E:\SIH\SIH_Results\active_learning_saad\domain_selection_summary.csv` | 1995 | `b210d9e807708baf46b6a099e18764db634e042dda1f7ebff8e5061ec866ac80` |
| `E:\SIH\SIH_Results\active_learning_saad\random_100.csv` | 47064 | `ae36aa0eae9a20d426081f96eca17525b788ad679f29d05dfc62bb0c497ffc3b` |
| `E:\SIH\SIH_Results\active_learning_saad\random_250.csv` | 117278 | `fe9c4843a35fa21d6c4a3ec9ff35960b582cfc02d9c49dd475a663705c208041` |
| `E:\SIH\SIH_Results\active_learning_saad\random_500.csv` | 233552 | `bc831cd528b7c82c0cdb114067f6938917c50433e768ddb2c790ee2618533ee9` |
| `E:\SIH\SIH_Results\active_learning_saad\saad_uncertainty_100.csv` | 38005 | `2017612bf8dfeb2091b87a89bd5fcc6af52f4c3eaac18bd8e99080a4dd5a8481` |
| `E:\SIH\SIH_Results\active_learning_saad\saad_uncertainty_250.csv` | 104123 | `ed3147e8fb9188d24b4d6733fdea03bd5f76804896af19f39ca6184de7227ccd` |
| `E:\SIH\SIH_Results\active_learning_saad\saad_uncertainty_500.csv` | 220534 | `eb8c0269ee49e72f60d188fa94523d04609024b1db7bb68ad800c468090141a3` |
| `E:\SIH\SIH_Results\active_learning_saad\strategy_overlap.csv` | 1127 | `c6d113f4331b01e5a67bd2874afc00f828e4ce088350b172d070669b39b892b6` |
| `E:\SIH\SIH_Results\active_learning_saad\strategy_summary.csv` | 2079 | `27fde97259db7b18b0ffb086adec75fec039ede574a22b0dc2ed0a2512f11403` |
| `E:\SIH\SIH_Results\active_learning_saad\tta_uncertainty_100.csv` | 37050 | `456d6c7efbba7bf0e3a5560ba141571666467ae45841d7d6e87caa50395052a4` |
| `E:\SIH\SIH_Results\active_learning_saad\tta_uncertainty_250.csv` | 96502 | `1fc57ad79fc2582aca1c8e22e0be4c7e3465d6ffa18329733c9859ca515282c1` |
| `E:\SIH\SIH_Results\active_learning_saad\tta_uncertainty_500.csv` | 196019 | `f5a149611373c67e7bd0e98dab33aacca26b8c98cb93ccaf05e9ac5b30382a00` |
| `E:\SIH\SIH_Results\active_learning_saad\yolo_uncertainty_100.csv` | 38814 | `87806eadb9c2efb54bd70bf71338a8b700f26f9ae60777076306427fcc313f8c` |
| `E:\SIH\SIH_Results\active_learning_saad\yolo_uncertainty_250.csv` | 109795 | `98907a56737a05b7b8acbe19d1d90f68c998026d6695c89751d0b681537c07fe` |
| `E:\SIH\SIH_Results\active_learning_saad\yolo_uncertainty_500.csv` | 231400 | `aeaaa2738e77c4ca90638815f3994766f90c4d6c32f6f5c544af7ce5e8925ec2` |
| `E:\SIH\SIH_Results\calibration_cross_validation\calibration_cv_summary.txt` | 2361 | `5070bacb0e798631c27934578ed6751c62f1f8b320fbe2f6688def1873c03fa9` |
| `E:\SIH\SIH_Results\calibration_cross_validation\fold_results.csv` | 4692 | `e1aeffc9754a2d5e03a8e626e0d7d354a60733ce2d44bb03c36d56b345fff16a` |
| `E:\SIH\SIH_Results\calibration_cross_validation\oof_per_image_scores.csv` | 333973 | `ead15eb8a533f36d8179940221c5b68e4c8b4572c5b423a406e2a1da6a8f5edf` |
| `E:\SIH\SIH_Results\calibration_cross_validation\oof_results.csv` | 373 | `35bc86ef6ca0e27c869c0f45db02ebe737a96750e959a6e5e781aee093c42242` |
| `E:\SIH\SIH_Results\calibration_cross_validation\performance_summary.csv` | 952 | `3f31f73b687c41005aff79f188cc135e5b18683b35e924a05647963c68179995` |
| `E:\SIH\SIH_Results\calibration_cross_validation\weight_stability.csv` | 1300 | `a6de01f1608a8aabf45135cce4af03141c4932cc11a562bc03855814f98a1d43` |
| `E:\SIH\SIH_Results\calibration_cross_validation\weight_summary.csv` | 566 | `104ae12f032b0119ab94948a47881fdbf70f35c52ce00d7c66e8bd55b9001dff` |
| `E:\SIH\SIH_Results\calibration_leave_one_domain_out\fold_details\fold_1_AI4Shipwrecks.csv` | 1311 | `07be6a90e0385af3efe1fdeee1000ea7fb346c95271773890885ac9419ae820d` |
| `E:\SIH\SIH_Results\calibration_leave_one_domain_out\fold_details\fold_2_GhostVision.csv` | 1355 | `f6b6fcc3899da5e0dc4067a143ad72dc9cbd0001a2c48b9ed6d2b4dfc104f5ca` |
| `E:\SIH\SIH_Results\calibration_leave_one_domain_out\fold_details\fold_3_SubPipeMini2.csv` | 850 | `a36bca1f719e57f303d726d8667e4ed6ef4bd146179e3b7121f8f3464922bf0f` |
| `E:\SIH\SIH_Results\calibration_leave_one_domain_out\lodo_calibration_details.csv` | 1290 | `623658f83bd5d5493a0777923a41769d52bfae9e85dc4986c4bc019fc2a6fab0` |
| `E:\SIH\SIH_Results\calibration_leave_one_domain_out\lodo_domain_summary.csv` | 617 | `4fe095588b5852226afe5e499886ea8969fff7099cf85342384a9876a516cd24` |
| `E:\SIH\SIH_Results\calibration_leave_one_domain_out\lodo_fold_results.csv` | 2954 | `baa56cdd0e86b12e051b9f72c2d41788c493e840d293483032d0c39396f7ab25` |
| `E:\SIH\SIH_Results\calibration_leave_one_domain_out\lodo_performance_summary.csv` | 939 | `d99666eb989f36cd2e6c7822deaaf998f161b8f4ac84a5256ba4984d3d14cb69` |
| `E:\SIH\SIH_Results\calibration_leave_one_domain_out\lodo_summary.txt` | 2915 | `4a0c1106d01bad1cf2016e4444f7cee1e5ca6669eea0bf5dd2ac915f033f3b5a` |
| `E:\SIH\SIH_Results\calibration_leave_one_domain_out\lodo_weight_stability.csv` | 943 | `717665b700911e4def485cabe4875d7e1c770dd747e6f89d4eb6ad6c6d7c568d` |
| `E:\SIH\SIH_Results\calibration_leave_one_domain_out\lodo_weight_summary.csv` | 542 | `84892d877038f09ba72bf932e54657bd5ad5256bb638cb772988c6aeea254307` |
| `E:\SIH\SIH_Results\dataset_wise_eval\_eval_ai4shipwrecks\dataset.yaml` | 134 | `8af0ab9cd30d7d47a607ec3ae81a031d43dfbe5b2c786ba19f695ee968f69254` |
| `E:\SIH\SIH_Results\dataset_wise_eval\_eval_ghostvision\dataset.yaml` | 132 | `a4c151327de4b11a9d112151efb2de8c9138103476ace8ca81b324ec093cd708` |
| `E:\SIH\SIH_Results\dataset_wise_eval\_eval_subpipemini2\dataset.yaml` | 133 | `704e964504f728d8ee72842ec2b4ae346b0a36f776d96ef90a737a1ee39d04b4` |
| `E:\SIH\SIH_Results\dataset_wise_eval\_tmp_ai4shipwrecks\dataset.yaml` | 104 | `3be85e1217ade1d1173eb30f7894671d9f57f4f7657b8fb4f872015d1ea78128` |
| `E:\SIH\SIH_Results\dataset_wise_eval\_tmp_ghostvision\dataset.yaml` | 102 | `c8861346670b84032a30f391a9827a6947fa78f3feb15ebcd7e14790285a3b43` |
| `E:\SIH\SIH_Results\dataset_wise_eval\_tmp_subpipemini2\dataset.yaml` | 103 | `56a22e6b5e302676d26994d16b1c816be98fd58e6648da22e2f0f60b1c7a0260` |
| `E:\SIH\SIH_Results\dataset_wise_eval\AI4Shipwrecks_evaluation\predictions.json` | 3226378 | `997e77fafe3e17b8722ac480868dae17d302973e9d3b630affd3a587cc304da2` |
| `E:\SIH\SIH_Results\dataset_wise_eval\dataset_wise_results.csv` | 739 | `bcc4a100fd5541790607656389dfc789df378fb8032a53103b793a44250a7ac7` |
| `E:\SIH\SIH_Results\dataset_wise_eval\dataset_wise_results.json` | 1385 | `d6c11d084aaa8b6794489da05d073ec7c3f14ada33b85793b8828e59a76b9ce6` |
| `E:\SIH\SIH_Results\dataset_wise_eval\dataset_wise_summary.txt` | 1835 | `9b0a9e60d63f16bdff1021e7d34405966e9cd849e72a48fedc93af33086f5ec6` |
| `E:\SIH\SIH_Results\dataset_wise_eval\failure_analysis\confidence_collapse_cases.csv` | 164165 | `0771592bc0042dc6a1a9472699354619fba14bc3bc51c5ca9c445ee9aad75507` |
| `E:\SIH\SIH_Results\dataset_wise_eval\failure_analysis\FAILURE_ANALYSIS_REPORT.txt` | 3810 | `7dbce2e34da92ee8810e2256eb21e3f4f0df2a41cf03060a987084e29d6da419` |
| `E:\SIH\SIH_Results\dataset_wise_eval\failure_analysis\failure_analysis_summary.csv` | 3109 | `61c1df1eb4fec6048945733bfe29326ded7b69d9b937353d4b76f79d1f0c9e83` |
| `E:\SIH\SIH_Results\dataset_wise_eval\failure_analysis\failure_analysis_summary.json` | 7734 | `43f0d917e0a425fbd8e09b56e566f26938ba25677c5a80facfcc678636c8993e` |
| `E:\SIH\SIH_Results\dataset_wise_eval\failure_analysis\localization_collapse_cases.csv` | 303 | `7613b4f959ae59755e32019db496644fc79262d21c825a094ee4c9180afaee2b` |
| `E:\SIH\SIH_Results\dataset_wise_eval\failure_analysis\missing_prediction_images.csv` | 266 | `dd7b46e79d2bb8ad5d2ddab8aadce4c979ab501b5295eef0711f915cde17c698` |
| `E:\SIH\SIH_Results\dataset_wise_eval\failure_analysis\per_ground_truth_failure_analysis.csv` | 226276 | `ca031bcda922c26a4cd08d1acec4fd4ccf60fb707a65fc47f3a9f48dc5f13060` |
| `E:\SIH\SIH_Results\dataset_wise_eval\failure_analysis\per_image_failure_analysis.csv` | 390522 | `34a8750a0bcd81b0e2266c1395f512ee3a850185eb14ebf3820b991fb5955930` |
| `E:\SIH\SIH_Results\dataset_wise_eval\failure_analysis\worst_localized_ground_truths.csv` | 32329 | `0e4d8313cf4e960019a00e7db11f54c64206cf22da04d634aa0504e04d9c9841` |
| `E:\SIH\SIH_Results\dataset_wise_eval\GhostVision_evaluation\predictions.json` | 7749668 | `fea7c76d04b7e3cd0eff74b842d3824e3838ead8629c41e781ec509a01bcd576` |
| `E:\SIH\SIH_Results\dataset_wise_eval\SubPipeMini2_evaluation\predictions.json` | 114118 | `44bd03efaaca7ebf0ca63e6daf60e21ae59cb220f5be95b5db3b6e6ea950d137` |
| `E:\SIH\SIH_Results\final_saad_evaluation\domain_wise_test_metrics.csv` | 2614 | `c7494d44952972a507fe1267c32c51a7ebdb1360028344824a188317e6c3aa1b` |
| `E:\SIH\SIH_Results\final_saad_evaluation\final_normalization_parameters.csv` | 318 | `a5629456e6ee0dbfb340ae356bdee27fd2007c9b756eaff5b561a63aeee6900b` |
| `E:\SIH\SIH_Results\final_saad_evaluation\final_test_metrics.csv` | 1311 | `b4cdf3bc7c5fa06309fcef5eb28ddc75359551ae9f0ada4251880b65f98a87c8` |
| `E:\SIH\SIH_Results\final_saad_evaluation\final_test_scores.csv` | 289169 | `5952c3ea8e9acfc6c07308f0146f3de41f72cacee39615b8ab99dce6fbc44515` |
| `E:\SIH\SIH_Results\final_saad_evaluation\FINAL_VERDICT.txt` | 1223 | `9b12941a5efeb100bef642964a588324d3811bad8d9165ec8c32e1a45cdfd1dc` |
| `E:\SIH\SIH_Results\final_saad_evaluation\fixed_threshold_test_metrics.csv` | 2221 | `f0a4f1c29246a19c44d093da64d5f6b542e203ab2212ef5dff2c5235f19820a1` |
| `E:\SIH\SIH_Results\final_saad_evaluation\test_evidence_distributions.csv` | 1664 | `9a66f25357b27a62579f1e3e34345060ae0a6bc38cdb1d3829ec500395576d78` |
| `E:\SIH\SIH_Results\final_saad_evaluation\validation_frozen_scores.csv` | 558470 | `28cc2d90c8dc3342a87ac85658eb7ce971c6684460e2d39244d1ae82253a60d9` |
| `E:\SIH\SIH_Results\final_saad_evaluation\validation_normal_thresholds.csv` | 616 | `ef7fd2c60a636c9f09dd2a00c9d3807002c51a3ded7d5d3b68d73cab37adaedf` |
| `E:\SIH\SIH_Results\latent_normalizing_flow\best_flow.pt` | 3714807 | `dbfc83c3baf1ac3a840b708b2b02b22c6a0753f1c969d7b18d623efc2e40ac65` |
| `E:\SIH\SIH_Results\latent_normalizing_flow\evaluation\anthropogenic_image_scores.csv` | 186394 | `335accea72aa2b13201542944434a1dccead19f388b46f2361b7f66a551d4407` |
| `E:\SIH\SIH_Results\latent_normalizing_flow\evaluation\combined_scores.csv` | 133522 | `a6db4f69c81f76c259f473fd294218cd243d2b1ec457926bf31c05441d5a32d1` |
| `E:\SIH\SIH_Results\latent_normalizing_flow\evaluation\dataset_results.csv` | 703 | `6f3654632e7353af39c3707af029987d3af6b011681832aaa55a13be15a108dd` |
| `E:\SIH\SIH_Results\latent_normalizing_flow\evaluation\evaluation_summary.txt` | 1196 | `d55bfaa4a2b690a6af81a65c97047dfa8d61a26b5a6f1562926f1a7aad58275b` |
| `E:\SIH\SIH_Results\latent_normalizing_flow\evaluation\normal_scores.csv` | 549435 | `6b1d401ba48f4f73c238a033b21248d29da7e3b432788901a6a900ed98c7f1ff` |
| `E:\SIH\SIH_Results\latent_normalizing_flow\flow_summary.txt` | 1296 | `a33df9b7cd6cb0b29910e1942af6865ffe9a2f84a8dbc754dc1b996a1469c896` |
| `E:\SIH\SIH_Results\latent_normalizing_flow\latent_mean.pt` | 2117 | `56771caed4bbecc6ff019d4f63a3e16664ade7f196462c8ef223e031da8f351c` |
| `E:\SIH\SIH_Results\latent_normalizing_flow\latent_std.pt` | 2110 | `087742e906b695d06fa0a78fe7ddd31e3a02259d1569669fda0483273dbf3326` |
| `E:\SIH\SIH_Results\latent_normalizing_flow\test_mu.pt` | 722473 | `a9ab8000c8708592e634c17fcc1403ae9343686971fa034334d1382c1849b23a` |
| `E:\SIH\SIH_Results\latent_normalizing_flow\test_scores.pt` | 7237 | `b3aaa8830d901055a5b03462d247756fed681dc42c7f249f7fa0544d047536c6` |
| `E:\SIH\SIH_Results\latent_normalizing_flow\train_mu.pt` | 5876784 | `e822aa695c836950bf32720c24274c8c12f7edec237c87202f4de017f552cef4` |
| `E:\SIH\SIH_Results\latent_normalizing_flow\train_scores.pt` | 47500 | `8210f2ef8df390be9597898bf413fb749c05835913ac04c3762569dd753e9198` |
| `E:\SIH\SIH_Results\latent_normalizing_flow\training_history.csv` | 1817 | `1c81278a471cdf57b2fd64fda91ba9b5f9a7dbcad807563fd0f5f5578d1b8edb` |
| `E:\SIH\SIH_Results\latent_normalizing_flow\val_mu.pt` | 719842 | `324223dc8001d3042120ad89031d1f1786db796976b855272063c87fe57e8bc3` |
| `E:\SIH\SIH_Results\latent_normalizing_flow\val_scores.pt` | 7166 | `5bd34e0f70e72c1cc6589ac1601b5ec10f837cf38b454810a9acc4965c60c23c` |
| `E:\SIH\SIH_Results\mar_evaluation\mar_all_scores.csv` | 711865 | `e1575cbf4e3337b991383dc877e33a922743ec83c50cb5e63f745d0cc66ad7a8` |
| `E:\SIH\SIH_Results\mar_evaluation\mar_domain_metrics.csv` | 504 | `f894e32762926d48f583226ed6e012d9a19c91347053a0d5b1e7e09850f9a384` |
| `E:\SIH\SIH_Results\mar_evaluation\MAR_EVALUATION_SUMMARY.json` | 832 | `1dad89509be4a98ddcde4fc92371c12df9615aad49c36783b92a511ebc6c3f88` |
| `E:\SIH\SIH_Results\mar_evaluation\mar_metrics.csv` | 422 | `ee6c2117258e26dfcf1b0a1256ab2dc7a94cface73bef8d2797934790ec8a8dc` |
| `E:\SIH\SIH_Results\mar_evaluation\mar_score_distributions.csv` | 916 | `cb2c3e66f9016de5f0ad0b9ee8567856d5a192bb69f800b6a3484cb27c992739` |
| `E:\SIH\SIH_Results\mar_evaluation\normal_only_thresholds.csv` | 251 | `6c1f10a4190e921428e8d6a797a4ace51433c43bcc699e88f0300dfcdb6985be` |
| `E:\SIH\SIH_Results\mar_evaluation\scores\anthropogenic_mar_scores.csv` | 128768 | `c155bf2f4ffe083b4ab16c5a678ea0baea67152a065c7de684177b668e759844` |
| `E:\SIH\SIH_Results\mar_evaluation\scores\normal_mar_scores.csv` | 583181 | `19ff6e6a6c2ca01f6f01d7a2c0bd161b26208fbfb7208d7038e6b522b798d44f` |
| `E:\SIH\SIH_Results\mar_normal_seabed\checkpoints\best.pt` | 66969006 | `4febb7991c540081e6dd49c2b2a944e3ef25e5688e50f945b4410c6c28b4f114` |
| `E:\SIH\SIH_Results\mar_normal_seabed\mar_summary.json` | 671 | `eed6f38894463e7868042825313612849660557999b3d26208a21c9ee95af68d` |
| `E:\SIH\SIH_Results\mar_normal_seabed\MAR_TRAINING_SUMMARY.txt` | 675 | `a1e007864db2be2dce004950e073160573505b83cc0d686091b510b664b4071b` |
| `E:\SIH\SIH_Results\mar_normal_seabed\training_history.csv` | 7966 | `f34cd9f7fb489038c2845dac6c3ceb9472b91f76ffbcfee5f99e6fc399cfe8ef` |
| `E:\SIH\SIH_Results\roi_spatial_fusion\dataset_wise_roi_results.csv` | 4059 | `9ef39ac15c240dead541f49a581b10ea6ef580fd8ce7e571b313164f77cf9150` |
| `E:\SIH\SIH_Results\roi_spatial_fusion\roi_candidate_scores.csv` | 19208487 | `6eee0700c2e94acde2eedc9d5d1361a9b13910c1513256919e5f039276beb073` |
| `E:\SIH\SIH_Results\roi_spatial_fusion\roi_fusion_metrics.csv` | 1342 | `e7e8b42dac16f651f6edf6096e1b8c4eda331646a1b021e566e70874749a8140` |
| `E:\SIH\SIH_Results\roi_spatial_fusion\roi_fusion_summary.txt` | 3440 | `299d72b7e227aa2a54bab1ae89ddc8a0adf89d27bb65973733f736c2261e6649` |
| `E:\SIH\SIH_Results\roi_spatial_fusion\roi_vs_global_comparison.csv` | 781 | `2b14e8816d564dcd368dd46f8258cdaae5eab29c097f247ce75c629401f78448` |
| `E:\SIH\SIH_Results\roi_spatial_fusion\tp_fp_distributions.csv` | 1527 | `51bb189cc0408cae5596ee2e1cfccd7cbf1979c024f2c700a6ab4c82cd4301a6` |
| `E:\SIH\SIH_Results\saad_evidence_engine\class_wise_evidence_summary.csv` | 1060 | `79b816c50286cf9057232bf726d4f8051e4f5e1cc6d6a8e6c2538820d48862b7` |
| `E:\SIH\SIH_Results\saad_evidence_engine\dataset_wise_evidence_summary.csv` | 1744 | `608172ce9e5d6886d6f7884721ae127fc343c0056db592e286dd673c7d0cee16` |
| `E:\SIH\SIH_Results\saad_evidence_engine\decision_summary.csv` | 585 | `bba7e04536d7d5a459879b8a707b437871b6da4fc6ab5f60d9bc2dd555598751` |
| `E:\SIH\SIH_Results\saad_evidence_engine\decision_thresholds.csv` | 206 | `20b970350eb53aadfed1685bba730c0dfb7ffbeb7bf529a7b61f57dcc6e0f8ed` |
| `E:\SIH\SIH_Results\saad_evidence_engine\evidence_scores_isotonic.csv` | 402144 | `845dda6e5a3886ed71d233f53025f97ed535eb29aa0adaa4f870f590887b6ae7` |
| `E:\SIH\SIH_Results\saad_evidence_engine\evidence_scores_percentile.csv` | 396641 | `2cc19b2569c9bb865716008a9d1f33ee5def68cae8273abf81bbac018ca237b8` |
| `E:\SIH\SIH_Results\saad_evidence_engine\evidence_scores_platt.csv` | 418699 | `056f13dcf53fdc40ed3a6e5926d91ae40808aa392ce4d8d19033c9183c8aff8d` |
| `E:\SIH\SIH_Results\saad_evidence_engine\ranked_review_candidates.csv` | 418699 | `cd05d36efb7105f60a867a1fc759cca8ee94d749aaadbb7c9b0560fe1730ac4d` |
| `E:\SIH\SIH_Results\saad_evidence_engine\saad_evidence_engine_summary.txt` | 4956 | `d03027d7fe2987ea3e5284218600a68ec85c3ff8c107c5488473b8c01c0b82ba` |
| `E:\SIH\SIH_Results\saad_evidence_engine\saad_primary_evidence_scores.csv` | 418699 | `056f13dcf53fdc40ed3a6e5926d91ae40808aa392ce4d8d19033c9183c8aff8d` |
| `E:\SIH\SIH_Results\saad_evidence_engine\top_200_uncertain_candidates.csv` | 71642 | `b80c229227d41feb218d4059fe90b1502c220f9ad3a4b6bea5b5e25e1c9c5792` |
| `E:\SIH\SIH_Results\saad_evidence_engine_v2\dataset_wise_test_metrics.csv` | 1561 | `708ae74c0cdac67c950939c8fd0b7b7b4d41e0ab17c6d99e5cad322aef83e661` |
| `E:\SIH\SIH_Results\saad_evidence_engine_v2\engine_config.json` | 689 | `e7fca995ba9e69781f84576a755c1e4103a64577a233f6abec48dd72ba5100e1` |
| `E:\SIH\SIH_Results\saad_evidence_engine_v2\EVIDENCE_ENGINE_REPORT.txt` | 1866 | `2b1592015c84d483d1795dbf6952a7c1436fbeffb1c5445c01bed50f7901cdc0` |
| `E:\SIH\SIH_Results\saad_evidence_engine_v2\normalization_parameters.csv` | 334 | `941c8c383f6c7734825a6e7c215f015b7762690fdce624545a042a3e0731cd75` |
| `E:\SIH\SIH_Results\saad_evidence_engine_v2\posthoc_test_metrics.csv` | 671 | `2c868d49105676eedd6aa055411e2f4c452d1365636a2b2452de7a8be3d6dae8` |
| `E:\SIH\SIH_Results\saad_evidence_engine_v2\test_action_summary.csv` | 276 | `2f914596aaf5310bc4e4fb6215d47c32e5005d2819d050a442432d03deb5a3f0` |
| `E:\SIH\SIH_Results\saad_evidence_engine_v2\test_evidence_queue.csv` | 260607 | `39a51ec8061ce804088f701f8cd7b8d78dc5a39ef2268b47d4cb20d199b46448` |
| `E:\SIH\SIH_Results\saad_evidence_engine_v2\test_profile_summary.csv` | 937 | `e75e0f542ec59f5b2d79e475924b530441de7e50e38012b5725f88a8d17ab95d` |
| `E:\SIH\SIH_Results\saad_evidence_engine_v2\top_review_queue.csv` | 46000 | `c5019676dfbe36041643a67f1b13175f0bab01959a98826dc3e295df6b4f9c55` |
| `E:\SIH\SIH_Results\saad_evidence_engine_v2\validation_action_summary.csv` | 276 | `5f0ac12bc190a148750362042a6cbaee56972b5dbd851e12212f76f07601d6ae` |
| `E:\SIH\SIH_Results\saad_evidence_engine_v2\validation_evidence_queue.csv` | 500316 | `749450ef5e00677cd17065dce7cd7149f8dee759b89f4bf79f30b3918fa68be8` |
| `E:\SIH\SIH_Results\saad_evidence_engine_v2\validation_profile_summary.csv` | 925 | `401f29f6ce482188b46bbe9c188815faf8a80f32687bef129e9341f52edc9b79` |
| `E:\SIH\SIH_Results\saad_evidence_engine_v3\dataset_wise_test_metrics.csv` | 2135 | `c0e4ed3590278013308679839f21b49475c93c39a2605685ea682e14f3d7ee7a` |
| `E:\SIH\SIH_Results\saad_evidence_engine_v3\engine_config.json` | 979 | `24d920367c2f6c25ccc6439d40e542fab4072935db94708afd7355dadc418170` |
| `E:\SIH\SIH_Results\saad_evidence_engine_v3\EVIDENCE_ENGINE_V3_REPORT.txt` | 2591 | `ec1d3181cda4fc6a91c83381dc3db231e2aeda3088599d524babd77d3e0cb230` |
| `E:\SIH\SIH_Results\saad_evidence_engine_v3\HITL_review_queue.csv` | 402975 | `bfaa6d5e1390757d25d1212a90f1eab7eefe6ebfccdb7e05743fcfdb4fbc6823` |
| `E:\SIH\SIH_Results\saad_evidence_engine_v3\normalization_parameters.csv` | 334 | `941c8c383f6c7734825a6e7c215f015b7762690fdce624545a042a3e0731cd75` |
| `E:\SIH\SIH_Results\saad_evidence_engine_v3\posthoc_test_metrics.csv` | 894 | `6fb82718587a562b40700ab2f87866d7f54e32c3c29a3845bd9b50177c500bb8` |
| `E:\SIH\SIH_Results\saad_evidence_engine_v3\test_action_summary.csv` | 610 | `1696f20cb700a1990c18d06af318260ab4eb03ce1349e27cb19940bf4b86b3c2` |
| `E:\SIH\SIH_Results\saad_evidence_engine_v3\test_evidence_queue.csv` | 404069 | `52f7bd65f6c22a283e38a003192b3381f9ccb5e25b8b21bacf3e20b1dd2718be` |
| `E:\SIH\SIH_Results\saad_evidence_engine_v3\test_profile_summary.csv` | 1200 | `fcadbbd6af13744e6220de9c84cd8cea878a5cc19a9a0355bff1b4c235542c22` |
| `E:\SIH\SIH_Results\saad_evidence_engine_v3\top_100_HITL_queue.csv` | 72581 | `8701aeb9a88a407e8d92f12fb1ee9987b44fb6253d7326d47f3e11bd07cc4731` |
| `E:\SIH\SIH_Results\saad_evidence_engine_v3\uncertainty_queue.csv` | 319794 | `b65d7fe65b5f0215876b6577d6f8cfcc40bf317f69f21b36c0850b5d394f397b` |
| `E:\SIH\SIH_Results\saad_evidence_engine_v3\validation_action_summary.csv` | 541 | `2f0ff1f4ee01f2206e575ff5f85e8582bf0d15344fa836b20f181dd2f2f47d51` |
| `E:\SIH\SIH_Results\saad_evidence_engine_v3\validation_evidence_queue.csv` | 785763 | `15a4af3a193548f1b91803645fc23e3d26d5c85f16ed7bf5eff2d3a7577d99bf` |
| `E:\SIH\SIH_Results\saad_evidence_engine_v3\validation_profile_summary.csv` | 1207 | `ab1a4bfc1225dd435f3c3e1df17fdcfc9b13c5896b0d7807333c09a7b42b0bba` |
| `E:\SIH\SIH_Results\saad_evidence_ranking\class_wise_distributions.csv` | 2440 | `edfe309195803e6f768f40e382e14cbc4b4dcc668f5fd088515fe4a44961b99d` |
| `E:\SIH\SIH_Results\saad_evidence_ranking\dataset_wise_policy_results.csv` | 3584 | `3e9d5daf19ed0212e40cb8f14b6a930ba995c1d088d497e23ab82dbe2a188087` |
| `E:\SIH\SIH_Results\saad_evidence_ranking\evidence_profile_summary.csv` | 378 | `9197278d3c3e44e8dd528340d29ba868c58c1e0649fb12d9eb16589ea9a3294b` |
| `E:\SIH\SIH_Results\saad_evidence_ranking\evidence_ranking_summary.txt` | 5964 | `fa66f654998df57ef2b407c5b6f20244f0a394c92de2ee239c10d94729b5eaa3` |
| `E:\SIH\SIH_Results\saad_evidence_ranking\normalization_parameters.csv` | 165 | `d4fe06562e3b8ff940750eebaaf84a263bfbfa9a3b85df7c45a3b15aa7d8d431` |
| `E:\SIH\SIH_Results\saad_evidence_ranking\overall_policy_results.csv` | 1439 | `a72bbcb653379cf8c6c618a78f782505b4d1dd8a764b718abcb4c5d4c46552c4` |
| `E:\SIH\SIH_Results\saad_evidence_ranking\per_image_evidence_ranking.csv` | 403290 | `67969ce04ddc1cc857b76f862f9cf356cf297aedb6a876bce000e5f4a1ec34ba` |
| `E:\SIH\SIH_Results\saad_evidence_ranking\signal_agreement_summary.csv` | 532 | `b33e6c547db008cbafa0f8fe9cd0a93b8a1874bd91d566d3db77f96fb7f2d2b4` |
| `E:\SIH\SIH_Results\saad_fusion_evaluation\dataset_wise_scores.csv` | 1365 | `9719b95168f4c7158aa873291c0a231d62657129e8a03e74f1f934dd368b3449` |
| `E:\SIH\SIH_Results\saad_fusion_evaluation\fusion_results.csv` | 947 | `89f9a5756cd9e6715863daaf71f4eebc9bce4c1b41e895ad5054ec71768066dc` |
| `E:\SIH\SIH_Results\saad_fusion_evaluation\fusion_summary.txt` | 1680 | `82c3d03501af914b0236686f75bb0b8712a7f58532cb75a8a364a9ddd0c19d27` |
| `E:\SIH\SIH_Results\saad_fusion_evaluation\per_image_fusion_scores.csv` | 557122 | `b1487a4073fbcdd4a3afa6956ddffa7306770cf1720d4d251a20b2e1b7774466` |
| `E:\SIH\SIH_Results\saad_fusion_evaluation\threshold_results.csv` | 659 | `8ed6d9c81b574468cf890f3080cbd8b756a8f1b5d5083485688f28e3682276a4` |
| `E:\SIH\SIH_Results\score_calibration\all_weight_search_results.csv` | 55367 | `2cff577089e97a3c48aca3b52db391230931fd1146599703ae0268cfd1e0bdec` |
| `E:\SIH\SIH_Results\score_calibration\calibrated_fusion_results.csv` | 1164 | `0fa32409ece638b9e64de2c11211e046cd015938243790da8148dc2f6834dc19` |
| `E:\SIH\SIH_Results\score_calibration\calibration_summary.txt` | 1766 | `f7aa627031c717419d1a1a917c68b7eec54022030700c36c8774471775f5276e` |
| `E:\SIH\SIH_Results\score_calibration\dataset_wise_calibration_results.csv` | 582 | `e2c5f24f88be90e0d20c09911615e1dfcb6071df67867f200aa94183c5de833c` |
| `E:\SIH\SIH_Results\score_calibration\individual_calibration_results.csv` | 1239 | `6ae7f17911d94fd368f8f4649fd72894ed5e37a5e36dd0e70805afff6b7394ef` |
| `E:\SIH\SIH_Results\score_calibration\per_image_calibrated_scores.csv` | 328829 | `705abfc6f65f1f8e2c89c5f74a2f7bffe4f1e3b7c879acfd9d8e9be865b7f968` |
| `E:\SIH\SIH_Results\shadow_plausibility\dataset_wise_shadow_scores.csv` | 439 | `8c893303c05641770a11398922f5c8186620b9b76065c15e039300515a49d77a` |
| `E:\SIH\SIH_Results\shadow_plausibility\shadow_candidate_scores.csv` | 9354587 | `10990c9eb206742b6661827176abc0345cf2146945d13ddff1e7a351075e7a9b` |
| `E:\SIH\SIH_Results\shadow_plausibility\shadow_summary.txt` | 797 | `0ab2919511ef1d2cb41febb2d9379f77434078236eb4837b427d4262554e823b` |
| `E:\SIH\SIH_Results\shadow_plausibility\shadow_thresholds.csv` | 8456 | `c1920a6ad4ee70b12cbd9b248a6d3ca9aa012e4551337ed2a04724d684a466f9` |
| `E:\SIH\SIH_Results\vae_anomaly_evaluation\all_anomaly_scores.csv` | 647492 | `11960437ca21ee08e0dd93fd913d1f22ed61f5b0dba15f2576473b097f692282` |
| `E:\SIH\SIH_Results\vae_anomaly_evaluation\dataset_results.csv` | 332 | `abf6d9d7fe8dc077c33b507a5ba9cbc34b7c462818e469deb30702b715d9ac59` |
| `E:\SIH\SIH_Results\vae_anomaly_evaluation\evaluation_summary.txt` | 787 | `5921b5e14a94c48a8e069a4385661114720dd7f333688407d4b80e1e288c40e1` |
| `E:\SIH\SIH_Results\vae_normal_seabed\checkpoints\best.pt` | 218365942 | `b950f60ce1358b72a9b2c58f1fd0ec955b8e7f8850205662d976356f87e043c9` |
| `E:\SIH\SIH_Results\vae_normal_seabed\checkpoints\last.pt` | 218365942 | `1ca4ffdc127ecc7b83967bbb092f1fa8ee2954e3ba5a6445d5343714cf9b1344` |
| `E:\SIH\SIH_Results\vae_normal_seabed\test_metrics.txt` | 154 | `346677758a705688e671ae7e4b951e12bb4c1c07486d9a9c417884f5dc6d1aee` |
| `E:\SIH\SIH_Results\vae_normal_seabed\training_history.csv` | 6721 | `dad743165d1e38354917275579c654cba622d8f1fb572141de97a41d6b8df594` |
| `E:\SIH\SIH_Results\vae_normal_seabed_v2_ssim\checkpoints\best.pt` | 218366582 | `2dcd2a42cada1f5b35298e5f973e295dda5623d587effe6002489c9d0b527a17` |
| `E:\SIH\SIH_Results\vae_normal_seabed_v2_ssim\checkpoints\last.pt` | 218366582 | `2b4a0ac1d9c143d019aee72d31ae46380b2dbb8305d6965b532cf6e9bfaa57e4` |
| `E:\SIH\SIH_Results\vae_normal_seabed_v2_ssim\test_metrics.txt` | 405 | `d5c78eb930e481033a0d28bd27037e0d35ed29e0b4043d73d3bbc4c7ccd5968d` |
| `E:\SIH\SIH_Results\vae_normal_seabed_v2_ssim\training_history.csv` | 3861 | `4b697f78d4571aaa88ce4cae1cdfb405d8aac1f9289fdee14acc474cfe356894` |
| `E:\SIH\SIH_Results\validation_weighted_fusion\dataset_wise_weighted_results.csv` | 339 | `42d0f3c2c561f6c10bff75a69a02e6e8da12a1b31efe82e6589077de9863147f` |
| `E:\SIH\SIH_Results\validation_weighted_fusion\fusion_comparison.csv` | 305 | `67f304c7d5c971550c21a8d9ea5f36388d9d079c25cee2b44a7b013b2abbb5f5` |
| `E:\SIH\SIH_Results\validation_weighted_fusion\per_image_weighted_scores.csv` | 227995 | `bdf591e7d3c215c2f55c7c919d834b5b04e4f354779e1b12444b205030ac0a1e` |
| `E:\SIH\SIH_Results\validation_weighted_fusion\test_raw_scores.csv` | 181098 | `25a0041af26919469dc76c05d6ea27902aaaf00c5b8caf2e6658175e33852f07` |
| `E:\SIH\SIH_Results\validation_weighted_fusion\validation_raw_scores.csv` | 337469 | `558b7eee5103a2af6aed7c3ae9c3329f67e509c18f035727d9849e5be2674b6b` |
| `E:\SIH\SIH_Results\validation_weighted_fusion\weight_search_results.csv` | 16481 | `b081c38c749dfb31646730cfca23cb8819777e3e40d43d34180ec7cc643213fb` |
| `E:\SIH\SIH_Results\validation_weighted_fusion\weighted_fusion_summary.txt` | 1361 | `15303f6a4527f6a5631a0830103f3e4f30e8ea7dd9e3e54df4c4eb9007e27e10` |
| `E:\SIH\SIH_Results\yolo26s_generic_baseline\args.yaml` | 1851 | `b2a80725099eb996d9d405eef1fe9153894a9dcc1ac5aedcec78d6b890a4fad2` |
| `E:\SIH\SIH_Results\yolo26s_generic_baseline\results.csv` | 3728 | `d49601c3294528789a0ab36f63b61e28253a13b8b49b33fa68b350a5e881e8d0` |
| `E:\SIH\SIH_Results\yolo26s_generic_baseline\weights\best.pt` | 20303429 | `e069ee7b5180bf35e7f71f908f47160ab9e70266d611374dd77e0a34e741eb8e` |
| `E:\SIH\SIH_Results\yolo26s_generic_baseline\weights\last.pt` | 20303429 | `c3f4941973ff52d65d0c43b98cad7dcb807b5f5a05e71df301e21d418515e9d9` |
| `E:\SIH\SIH_Results\yolo_tta_validation\tta_image_level_metrics.csv` | 869 | `08575de67fe8e6e646da32afe4dabd58d33be4a2587db17f125d3e51035e3166` |
| `E:\SIH\SIH_Results\yolo_tta_validation\tta_image_scores.csv` | 372422 | `7e748c8305339b54015d346646ee99c5abf596c7b4e65640e0fa871528e2d372` |
| `E:\SIH\SIH_Results\yolo_tta_validation\tta_variant_scores.csv` | 1628033 | `394812ad6e9f4f4fe8d33644f9162fcd7c49820be33789c6d64c51836c170bd2` |
| `E:\SIH\SIH_Results\yolo_tta_validation\tta_variant_summary.csv` | 1041 | `1a59e6641cc518e0bd241f0c7ca041827908e326cc0ebee183cbb14d9fa53918` |
| `E:\SIH\weights\yolo26n.pt` | 5544453 | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| `E:\SIH\yolo26s.pt` | 20422725 | `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b` |

Total hashed artifacts and metadata: **197**.
