
# replace the return statement in models/inbetweener_with_mask_with_spec.py,
# in InbetweenerTM.forward(), in the else: clause of "if 'motion0' in data and 'motion1' in data:",
# with the return statement below:

            return {
                'keypoints0t': kpt0t,
                'keypoints1t': kpt1t,
                'vb0': (vb0 > 0).float(),
                'vb1': (vb1 > 0).float(),
                'r0': motion_output0,
                'r1': motion_output1,
                'loss': -1,
                'skip_train': True,
            }

# replace the "tqdm" loop in DraftRefine.eval with this to run on a given pair of png,json:

            import custom_data
            import visualize_custom
            # this works with .png and .json files from the paper's data set - you can replace with your own
            s0 = custom_data.DataSample('data/ml100_norm/all/frames/chip_abe/Image0001.png', 'data/ml100_norm/all/labels/chip_abe/Line0001.json')
            s1 = custom_data.DataSample('data/ml100_norm/all/frames/chip_abe/Image0005.png', 'data/ml100_norm/all/labels/chip_abe/Line0005.json')
            data = custom_data.make_model_input(s0, s1)
            
            pred = model(data)
            for k, v in pred.items():
                pred[k] = v
                pred = {**pred, **data}

            img_vis = visualize_custom.visualize_custom(pred)
            cv2.imwrite('out.png', img_vis)

