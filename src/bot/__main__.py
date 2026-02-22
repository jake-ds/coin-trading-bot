"""Allow running as `python -m bot.main`."""

import asyncio

from bot.main import main

asyncio.run(main())
