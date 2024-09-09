#!/bin/bash

py test_model.py nogam_frac &
py test_model.py signal_frac &
py test_model.py nogam_frac5 &
py test_model.py signal_frac5 &
py test_model.py nogam_deduct5 &
py test_model.py signal_deduct5 &
py test_model.py signal_1gam &
py test_model.py nogam_1gam &
py test_model.py signal_sim &
py test_model.py nogam_sim &
wait


