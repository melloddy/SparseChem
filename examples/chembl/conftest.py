def pytest_addoption(parser):
    parser.addoption("--dev", action="store", default="cuda:0", help="Pytorch device (e.g., cuda:0, cpu)")

def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.dev
    if 'dev' in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("dev", [option_value])


